import copy
import csv
import os
import random

import numpy as np
import torch
import wandb
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
from transformers import AdamW

from info_nce import InfoNCE
from models.model import Model
from utils.constants import DEVICE
from utils.metric_utils import cosine_similarity_matrix_2
from utils.results_utils import make_dir, calculate_cebab_score


class MatchingRepresentation(Model):

    def __init__(self, tokenizer, pretrained_model=None, pretrained_model_path=None, model_description=None,
                 examples_per_pair=5):
        self.pretrained_model_path = pretrained_model_path
        self.pretrained_model = pretrained_model
        self.tokenizer = tokenizer
        self.pretrained_model_path = pretrained_model_path
        self.examples_per_pair = examples_per_pair
        if self.pretrained_model is None and self.pretrained_model_path is None:
            raise ValueError('Either pretrained_model or pretrained_model_path must be provided')
        if self.pretrained_model is not None and self.pretrained_model_path is not None:
            raise ValueError('Only one of pretrained_model or pretrained_model_path must be provided')
        self.flag_train = True
        if pretrained_model is None and pretrained_model_path is not None:
            self.flag_train = False
            self.pretrained_model = torch.load(self.pretrained_model_path)
        self.lm_model = copy.deepcopy(self.pretrained_model)

        super().__init__(pretrained_model_path=self.pretrained_model_path, model_description=model_description)

    def set_lm_model(self, model):
        self.lm_model = model

    def get_representation_model(self):
        return self.lm_model

    def get_lm_model(self):
        return self.pretrained_model

    def train(self, config, wandb_on, train_batch, mean_loss, results_dir=None, valid_batch=None, test_batch=None,
              cebab_eval=None, loss_version=0):
        # batches = {'train': train_batch, 'validation': valid_batch, 'test': test_batch}
        if cebab_eval is None:
            cebab_eval = {'model_to_explain': None, 'explainers': None,
                          'pairs_validation': None, 'pairs_test': None,
                          'matching_description': None}
        else:
            cebab_eval = {'model_to_explain': cebab_eval['model_to_explain'], 'explainers': cebab_eval['explainers'],
                          'pairs_validation': cebab_eval['pairs_validation'], 'pairs_test': cebab_eval['pairs_test'],
                          'matching_description': cebab_eval['matching_description']}
        if loss_version == 0:
            coefficients = {
                'tcf_cfc': config.tcf_cfc,
                'tcf_pax': config.tcf_pax,
                'tcf_nax': config.tcf_nax,
                'pax_nax': config.pax_nax,
                'pax_cfc': config.pax_cfc,
                'cfc_nax': config.cfc_nax,
            }


        elif loss_version == 1:
            coefficients = {'approx': config.approx, 'c_cf': config.c_cf,
                            't_cf': config.t_cf, 'negative': config.negative}
        else:
            raise ValueError('Loss version not supported')
        rep_model = self.get_representation_model()

        rep_model.train()
        optimizer = AdamW(rep_model.parameters(), lr=config.learning_rate)
        loss = InfoNCE(negative_mode='unpaired', temperature=config.temperature,
                       coefficients=coefficients, device=DEVICE, mean_loss=mean_loss, loss_version=loss_version)

        best_model = None
        best_eval_loss = np.inf
        best_epoch = 0

        # flag_approx = True if coefficients['approx'] != 0 else False
        # flag_c_cf = True if coefficients['c_cf'] != 0 else False
        # flag_t_cf = True if coefficients['t_cf'] != 0 else False
        # flag_negative = True if coefficients['negative'] != 0 else False

        flag_approx = True
        flag_c_cf = True
        flag_t_cf = True
        flag_negative = True
        rep_model.to(DEVICE)

        for epoch in range(config.epochs):
            progress_bar = tqdm(train_batch)
            for i, t_batch in enumerate(progress_bar):
                query = t_batch['query']
                t_cf = t_batch['t_cf']
                c_cf = t_batch['c_cf']
                negative = t_batch['negative']
                approx = t_batch['approx']

                t_cf_embeddings = None
                negative_embeddings = None
                c_cf_embeddings = None
                approx_embeddings = None

                query_embedding = rep_model(input_ids=query['input_ids'].unsqueeze(dim=0).to(DEVICE),
                                            attention_mask=query['attention_mask'].unsqueeze(dim=0).to(DEVICE),
                                            token_type_ids=query['token_type_ids'].unsqueeze(dim=0).to(
                                                DEVICE)).pooler_output
                if flag_t_cf:
                    t_cf_input_ids = torch.stack(list(t_cf['input_ids']))
                    t_cf_attention_mask = torch.stack(list(t_cf['attention_mask']))
                    t_cf_token_type_ids = torch.stack(list(t_cf['token_type_ids']))

                    if self.examples_per_pair < len(t_cf['input_ids']):
                        example_idx = random.sample(range(len(t_cf['input_ids'])),
                                                    self.examples_per_pair)
                        t_cf_input_ids = t_cf_input_ids[example_idx]
                        t_cf_attention_mask = t_cf_attention_mask[example_idx]
                        t_cf_token_type_ids = t_cf_token_type_ids[example_idx]

                    t_cf_embeddings = rep_model(input_ids=t_cf_input_ids.to(DEVICE),
                                                attention_mask=t_cf_attention_mask.to(DEVICE),
                                                token_type_ids=t_cf_token_type_ids.to(DEVICE)).pooler_output

                if flag_negative:
                    negative_input_ds = torch.stack(list(negative['input_ids']))
                    negative_attention_mask = torch.stack(list(negative['attention_mask']))
                    negative_token_type_ids = torch.stack(list(negative['token_type_ids']))

                    if self.examples_per_pair < len(negative['input_ids']):
                        example_idx = random.sample(range(len(negative['input_ids'])),
                                                    self.examples_per_pair)
                        neg_input_ids = negative_input_ds[example_idx]
                        neg_attention_mask = negative_attention_mask[example_idx]
                        neg_token_type_ids = negative_token_type_ids[example_idx]

                    negative_embeddings = rep_model(input_ids=neg_input_ids.to(DEVICE),
                                                    attention_mask=neg_attention_mask.to(DEVICE),
                                                    token_type_ids=neg_token_type_ids.to(DEVICE)).pooler_output

                if flag_c_cf:

                    c_cf_input_ids = torch.stack(list(c_cf['input_ids']))
                    c_cf_attention_mask = torch.stack(list(c_cf['attention_mask']))
                    c_cf_token_type_ids = torch.stack(list(c_cf['token_type_ids']))

                    if self.examples_per_pair < len(c_cf['input_ids']):
                        example_idx = random.sample(range(len(c_cf['input_ids'])),
                                                    self.examples_per_pair)
                        c_cf_input_ids = c_cf_input_ids[example_idx]
                        c_cf_attention_mask = c_cf_attention_mask[example_idx]
                        c_cf_token_type_ids = c_cf_token_type_ids[example_idx]

                    c_cf_embeddings = rep_model(input_ids=c_cf_input_ids.to(DEVICE),
                                                attention_mask=c_cf_attention_mask.to(DEVICE),
                                                token_type_ids=c_cf_token_type_ids.to(DEVICE)).pooler_output
                if flag_approx:
                    approx_input_ids = torch.stack(list(approx['input_ids']))
                    approx_attention_mask = torch.stack(list(approx['attention_mask']))
                    approx_token_type_ids = torch.stack(list(approx['token_type_ids']))

                    if self.examples_per_pair < len(approx['input_ids']):
                        example_idx = random.sample(range(len(approx['input_ids'])),
                                                    self.examples_per_pair)
                        approx_input_ids = approx_input_ids[example_idx]
                        approx_attention_mask = approx_attention_mask[example_idx]
                        approx_token_type_ids = approx_token_type_ids[example_idx]

                    approx_embeddings = rep_model(input_ids=approx_input_ids.to(DEVICE),
                                                  attention_mask=approx_attention_mask.to(DEVICE),
                                                  token_type_ids=approx_token_type_ids.to(DEVICE)).pooler_output

                outputs = loss(query=query_embedding, positive_keys=t_cf_embeddings,
                               negative_keys=negative_embeddings, confounder_counterfactual_keys=c_cf_embeddings,
                               approx_keys=approx_embeddings)

                # In case of gaining gradients from the previous batches, change this condition
                if True:
                    outputs.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                progress_bar.set_description(
                    f'Epoch {epoch + 1}/{config.epochs} | Loss: {outputs.item():.4f} |')

                if ((i + 1) % (len(train_batch) // 2)) == 0 or (i + 1 == len(progress_bar)):
                    print(f'starting evaluation, epoch = {epoch + 1}, iteration = {i}.')
                    rep_model.eval()
                    with torch.no_grad():
                        if valid_batch is not None:
                            loss_eval = evaluation(eval_set=valid_batch, model=rep_model, loss_function=loss,
                                                   device=DEVICE,
                                                   epoch=epoch, wandb_on=wandb_on, coefficients=coefficients,
                                                   examples_per_pair=self.examples_per_pair,
                                                   name='validation', path_dir=None, config=config, return_loss=True)
                            if best_eval_loss > loss_eval:
                                best_eval_loss = loss_eval
                                best_model = copy.deepcopy(rep_model)
                                best_epoch = epoch
                        else:
                            print('cebab eval is None')
                            best_model = rep_model
                            best_epoch = epoch

                        if (cebab_eval['model_to_explain'] is not None) and (
                                cebab_eval['explainers'] is not None) and \
                                (cebab_eval['pairs_validation'] is not None):
                            self.set_lm_model(rep_model)
                            for e in cebab_eval['explainers']:
                                e.set_representation_model(self)
                            log = calculate_cebab_score(model_to_explain=cebab_eval['model_to_explain'],
                                                        concepts=[config.treatment],
                                                        explainers=cebab_eval['explainers'],
                                                        pairs=cebab_eval['pairs_validation'],
                                                        name='validation',
                                                        wandb_on=wandb_on, return_log=True, path_dir=results_dir)
                            print(log)

                    rep_model.to(DEVICE)
                    rep_model.train()

                del t_batch
                torch.cuda.empty_cache()
        if results_dir:
            make_dir(results_dir)
            print(f'saving the model- best epoch{best_epoch}, best loss {best_eval_loss}')
            torch.save(best_model, os.path.join(results_dir, f'model.pt'))

        with torch.no_grad():
            if valid_batch is not None:
                if results_dir:
                    p_eval = os.path.join(results_dir, 'eval')
                    make_dir(p_eval)
                else:
                    p_eval = None
                # train_loss = calculate_loss(best_model, train_batches, loss)
                # evaluation(eval_set=train_batch, model=best_model, loss_function=loss, device=DEVICE,
                #            epoch=best_epoch, wandb_on=wandb_on, coefficients=coefficients,
                #            examples_per_pair=self.examples_per_pair,
                #            name='train', path_dir=p_eval, config=config, best=True)
                #
                # evaluation(eval_set=valid_batch, model=best_model, loss_function=loss, device=DEVICE,
                #            epoch=best_epoch, wandb_on=wandb_on, coefficients=coefficients,
                #            examples_per_pair=self.examples_per_pair,
                #            name='validation', path_dir=p_eval, config=config, best=True)
                #
                # if test_batch is not None:
                #     evaluation(test_batch, best_model, epoch=best_epoch,
                #                examples_per_pair=self.examples_per_pair,
                #                loss_function=loss, coefficients=coefficients, wandb_on=wandb_on,
                #                name='test', path_dir=p_eval, config=config, best=True)

                if cebab_eval['pairs_test'] is not None:
                    calculate_cebab_score(model_to_explain=cebab_eval['model_to_explain'],
                                          concepts=[config.treatment],
                                          explainers=cebab_eval['explainers'],
                                          pairs=cebab_eval['pairs_test'],
                                          name='test',
                                          wandb_on=wandb_on, return_log=False, path_dir=results_dir)

        self.set_lm_model(best_model.eval())
        print('return the model')
        return best_model.eval()

    def get_tokenizer(self):
        return self.tokenizer

    def get_model_description(self):
        return 'matching_representation'


def evaluation(eval_set, model, loss_function, wandb_on, device, epoch, name, coefficients, config, examples_per_pair,
               path_dir=None,
               return_loss=False, best=False):
    print('starting evaluation on the {} set'.format(name))
    # add here cebab score

    model.eval()
    pos_distances = []
    neg_distances = []
    approx_distances = []
    confounder_distances = []
    losses = []
    losses_fixed_c = []

    model = model.to(device)

    with torch.no_grad():
        for i, t_batch in enumerate(tqdm(eval_set)):

            query = t_batch['query']
            t_cf = t_batch['t_cf']
            c_cf = t_batch['c_cf']
            negative = t_batch['negative']
            approx = t_batch['approx']

            query_embedding = model(input_ids=query['input_ids'].unsqueeze(dim=0).to(DEVICE),
                                    attention_mask=query['attention_mask'].unsqueeze(dim=0).to(DEVICE),
                                    token_type_ids=query['token_type_ids'].unsqueeze(dim=0).to(
                                        DEVICE)).pooler_output

            t_cf_input_ids = torch.stack(list(t_cf['input_ids']))
            t_cf_attention_mask = torch.stack(list(t_cf['attention_mask']))
            t_cf_token_type_ids = torch.stack(list(t_cf['token_type_ids']))
            if examples_per_pair < len(t_cf['input_ids']):
                example_idx = random.sample(range(len(t_cf['input_ids'])),
                                            examples_per_pair)
                t_cf_input_ids = t_cf_input_ids[example_idx]
                t_cf_attention_mask = t_cf_attention_mask[example_idx]
                t_cf_token_type_ids = t_cf_token_type_ids[example_idx]

            t_cf_embeddings = model(input_ids=t_cf_input_ids.to(DEVICE),
                                    attention_mask=t_cf_attention_mask.to(DEVICE),
                                    token_type_ids=t_cf_token_type_ids.to(DEVICE)).pooler_output

            negative_input_ds = torch.stack(list(negative['input_ids']))
            negative_attention_mask = torch.stack(list(negative['attention_mask']))
            negative_token_type_ids = torch.stack(list(negative['token_type_ids']))

            if examples_per_pair < len(negative['input_ids']):
                example_idx = random.sample(range(len(negative['input_ids'])),
                                            examples_per_pair)
                neg_input_ids = negative_input_ds[example_idx]
                neg_attention_mask = negative_attention_mask[example_idx]
                neg_token_type_ids = negative_token_type_ids[example_idx]

            negative_embeddings = model(input_ids=neg_input_ids.to(DEVICE),
                                        attention_mask=neg_attention_mask.to(DEVICE),
                                        token_type_ids=neg_token_type_ids.to(DEVICE)).pooler_output

            c_cf_input_ids = torch.stack(list(c_cf['input_ids']))
            c_cf_attention_mask = torch.stack(list(c_cf['attention_mask']))
            c_cf_token_type_ids = torch.stack(list(c_cf['token_type_ids']))

            if examples_per_pair < len(c_cf['input_ids']):
                example_idx = random.sample(range(len(c_cf['input_ids'])),
                                            examples_per_pair)
                c_cf_input_ids = c_cf_input_ids[example_idx]
                c_cf_attention_mask = c_cf_attention_mask[example_idx]
                c_cf_token_type_ids = c_cf_token_type_ids[example_idx]

            c_cf_embeddings = model(input_ids=c_cf_input_ids.to(DEVICE),
                                    attention_mask=c_cf_attention_mask.to(DEVICE),
                                    token_type_ids=c_cf_token_type_ids.to(DEVICE)).pooler_output

            approx_input_ids = torch.stack(list(approx['input_ids']))
            approx_attention_mask = torch.stack(list(approx['attention_mask']))
            approx_token_type_ids = torch.stack(list(approx['token_type_ids']))

            if examples_per_pair < len(approx['input_ids']):
                example_idx = random.sample(range(len(approx['input_ids'])),
                                            examples_per_pair)
                approx_input_ids = approx_input_ids[example_idx]
                approx_attention_mask = approx_attention_mask[example_idx]
                approx_token_type_ids = approx_token_type_ids[example_idx]

            approx_embeddings = model(input_ids=approx_input_ids.to(DEVICE),
                                      attention_mask=approx_attention_mask.to(DEVICE),
                                      token_type_ids=approx_token_type_ids.to(DEVICE)).pooler_output

            losses.append(
                loss_function(query=query_embedding, positive_keys=t_cf_embeddings, negative_keys=negative_embeddings,
                              confounder_counterfactual_keys=c_cf_embeddings,
                              approx_keys=approx_embeddings).detach().cpu().numpy())
            losses_fixed_c.append(
                loss_function(query=query_embedding, positive_keys=t_cf_embeddings, negative_keys=negative_embeddings,
                              confounder_counterfactual_keys=c_cf_embeddings, approx_keys=approx_embeddings,
                              eval=True).detach().cpu().numpy())

            similarities = cosine_similarity_matrix_2(query_embedding, t_cf_embeddings).mean()
            pos_distances.append(np.mean(similarities))

            similarities = cosine_similarity_matrix_2(query_embedding, c_cf_embeddings).mean()
            confounder_distances.append(np.mean(similarities))

            similarities = cosine_similarity_matrix_2(query_embedding, approx_embeddings).mean()
            approx_distances.append(np.mean(similarities))

            similarities = cosine_similarity_matrix_2(query_embedding, negative_embeddings).mean()
            neg_distances.append(np.mean(similarities))

            del t_batch
            torch.cuda.empty_cache()

    model.cpu()

    pos_mean = np.mean(pos_distances)
    confounder_mean = np.mean(confounder_distances)
    approx_mean = np.mean(approx_distances)
    neg_mean = np.mean(neg_distances)

    differences_positive = pos_mean - approx_mean
    differences_negative = confounder_mean - neg_mean
    differences_samples = approx_mean - neg_mean
    differences_counterfactuals = pos_mean - confounder_mean

    is_order_relation = 0
    if differences_positive > 0 and differences_negative > 0 and differences_samples > 0 and differences_counterfactuals > 0:
        is_order_relation = 1

    if best:
        name = f'best_{name}'

    log = {f'treatment-counterfactual_{name}': round(pos_mean, 2),
           f'approx_{name}': round(approx_mean, 2),
           f'negative_{name}': round(neg_mean, 2),
           f'confounder-counterfactual_{name}': round(confounder_mean, 2),
           f'loss_{name}': round(np.mean(losses), 2),
           f'epoch_{name}': epoch,
           f'loss_fixed_coefficients_{name}': round(np.mean(losses_fixed_c), 2),
           f'differences_samples_{name}': round(differences_samples, 2),
           f'differences_counterfactuals_{name}': round(differences_counterfactuals, 2),
           f'differences_positive_{name}': round(differences_positive, 2),
           f'differences_negative_{name}': round(differences_negative, 2),
           f'is_order_relation_{name}': is_order_relation}

    if path_dir is not None:
        w = csv.writer(open(os.path.join(path_dir, f'{name}.csv'), "w"))
        for key, val in log.items():
            # write every key and value to file
            w.writerow([key, val])

    if wandb_on:
        # wandb.init()
        wandb.log(log)
    print(log)

    if return_loss:
        return round(np.mean(losses_fixed_c), 2)
