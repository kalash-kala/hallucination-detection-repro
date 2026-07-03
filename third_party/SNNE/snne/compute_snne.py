import os
import logging

import wandb
from tqdm import tqdm
import torch
import pandas as pd
import evaluate
from rouge_score import tokenizers
from sentence_transformers import SentenceTransformer

from snne.uncertainty.utils.eval_utils import auroc, auarc, aucpr, is_binary_list
from snne.uncertainty.utils import utils
from snne.uncertainty.utils.metric_utils import get_metric
from snne.uncertainty.uncertainty_measures.semantic_entropy import EntailmentDeberta, soft_nearest_neighbor_loss
from snne.uncertainty.utils.compute_utils import get_parser, setup_wandb, load_precomputed_results, collect_info, print_best_scores


# Set up log
utils.setup_logger()

# Parse arguments
args = get_parser()
logging.info("Args: %s", args)
utils.set_all_seeds(args.random_seed)

# Set up wandb
setup_wandb(args, prefix='compute_snne')

# Load pre-computed results
precomputed_results = load_precomputed_results(args)
validation_generations = precomputed_results['validation_generations']
save_embedding_path = precomputed_results['save_embedding_path']
save_dict = precomputed_results['save_dict']
lexsim_exist = precomputed_results['lexsim_exist']
list_semantic_ids = precomputed_results['list_semantic_ids']

# Load models
save_list = []
load_list = []

if lexsim_exist:
    load_list.append('lexsim')
else:
    save_list.append('lexsim')
if 'entail' in save_list:
    entailment_model = EntailmentDeberta()
else:
    entailment_model = None
if 'embedding' in save_list:
    if args.embedding_model == 'qwen':
        embedding_model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-7B-instruct", trust_remote_code=True)
        embedding_model.max_seq_length = 8192
    else:
        embedding_model = SentenceTransformer("Salesforce/SFR-Embedding-2_R")
else:
    embedding_model = None
tokenizer = tokenizers.DefaultTokenizer(use_stemmer=False).tokenize
rouge = evaluate.load('rouge', keep_in_memory=True)

if args.recompute_accuracy:
    # This is usually not enabled.
    logging.warning('Recompute accuracy enabled.')
    metric = get_metric(args.metric)
else:
    metric = None

# Collect info
result_dict = collect_info(
    args, 
    validation_generations, 
    metric, 
    entailment_model, 
    embedding_model, 
    rouge,
    tokenizer,
    list_semantic_ids,
    save_dict, 
    save_embedding_path, 
    save_list, 
    load_list
)

validation_is_true = result_dict['validation_is_true']
list_generation = result_dict['list_generation']
list_generation_log_likelihoods = result_dict['list_generation_log_likelihoods']
list_generation_lexcial_sim = result_dict['list_generation_lexcial_sim']

# Calculate SNN score
list_method_name = []
list_auroc = []
list_auarc = []
list_aucpr = []
list_temperature = []
list_variant = []
list_selfsim = []
list_similarity_name = []

validation_is_false = [1.0 - is_t for is_t in validation_is_true]
is_binary = is_binary_list(validation_is_false)
temperature_choice = [0.1, 1, 10, 100]
variant_choice = ['only_denom']
selfsim_choice = [True]
list_responses = [
    list_generation
]
list_similarity_matrix = [
    list_generation_lexcial_sim
]
similarity_name_choice = [
    'lexical_sim'
]

for variant in variant_choice:
    for selfsim in selfsim_choice:
        for temperature in temperature_choice:
            for response, similarity_matrix, similarity_name in zip(list_responses, list_similarity_matrix, similarity_name_choice):
                method_name_postfix = f'{variant}_temp{temperature}_selfsim{selfsim}_{similarity_name}-similarity'
                logging.info(method_name_postfix.center(100, '-'))
                list_snne = []
                list_wsnne = []
                
                for idx in tqdm(range(len(validation_is_true))):
                    # Compute SNN
                    snne = soft_nearest_neighbor_loss(
                        response[idx],
                        entailment_model, 
                        embedding_model, 
                        list_semantic_ids[idx],
                        similarity_matrix=similarity_matrix[idx],
                        variant=variant, 
                        temperature=temperature, 
                        exclude_diagonal=not selfsim).item()
                    list_snne.append(snne)
                    
                    # Compute WSNN
                    if args.compute_wsnn:
                        weight_pe = torch.exp(torch.tensor(list_generation_log_likelihoods[idx]))
                        weight_pe = weight_pe / weight_pe.mean()
                        wsnne = soft_nearest_neighbor_loss(
                            response[idx],
                            entailment_model, 
                            embedding_model, 
                            list_semantic_ids[idx],
                            similarity_matrix=similarity_matrix[idx],
                            variant=variant, 
                            temperature=temperature, 
                            exclude_diagonal=not selfsim,
                            weight=weight_pe).item()

                        list_wsnne.append(wsnne)

                # Collect AUROC score
                snne_choice = [list_snne, list_wsnne]
                list_snne_name = ['snne', 'wsnne']
                if not args.compute_wsnn:
                    snne_choice = snne_choice[:-1]
                    list_snne_name = list_snne_name[:-1]
                
                for snn, snne_name in zip(snne_choice, list_snne_name):
                    if is_binary:
                        snne_auroc = auroc(validation_is_false, snn)
                    else:
                        snne_auroc = -1
                    snne_auarc = auarc(snn, validation_is_true)
                    snne_aucpr = aucpr(snn, validation_is_true)
                    list_variant.append(variant)
                    list_selfsim.append(selfsim)
                    list_temperature.append(temperature)
                    list_similarity_name.append(similarity_name)
                    list_method_name.append(snne_name)
                    list_auroc.append(snne_auroc)
                    list_auarc.append(snne_auarc)
                    list_aucpr.append(snne_aucpr)
                
# Output to CSV
data_metrics = {
    'method': list_method_name,
    'variant': list_variant,
    'selfsim': list_selfsim,
    'temperature': list_temperature,
    'similarity': list_similarity_name,
    'auroc': list_auroc,
    'auarc': list_auarc,
    'prr': list_aucpr
}

df_metrics = pd.DataFrame(data_metrics)
# logging.info(df_metrics.head())

# Save results
os.makedirs('snne_results', exist_ok=True)
df_metrics.to_csv(f'snne_results/{args.dataset}_{args.model_name}_{args.num_generations}generations{args.suffix}_seed{args.random_seed}.csv', index=False)

# Print the best scores
print_best_scores(df_metrics, keyword='', list_scores=['auroc', 'auarc', 'prr'])

wandb.finish()