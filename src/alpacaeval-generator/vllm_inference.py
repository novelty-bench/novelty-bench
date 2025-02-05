from vllm import LLM, SamplingParams 
import torch
import os
import tqdm
import json
import pandas as pd 
import logging 
import sys
import argparse

SYSTEM_INSTRUCTION = 'Below is an instruction that is optionally paired with some additional context. Respond appropriately follows using the context (if any) \n'

def model_wise_formatter(prompts, model_identifier, system_instruction=SYSTEM_INSTRUCTION):
    if model_identifier == 'gemma':
        return [f"{system_instruction} \n ### USER: {prompt}\n ### ASSISTANT:" for prompt in prompts]
    if model_identifier == 'llama':
        return [f"{system_instruction} \n ### Instruction: {prompt} \n ### Response:" for prompt in prompts]
    if model_identifier == 'llama2':
        return [f"""<s> [INST] <<SYS>> {SYSTEM_INSTRUCTION} <</SYS>>{prompt} [/INST]""" for prompt in prompts]
    if model_identifier == 'gemma2': 
        return [f"""<bos><start_of_turn>user {SYSTEM_INSTRUCTION}<end_of_turn><start_of_turn>model {prompt}""" for prompt in prompts]
    else: 
        logging.error('ERROR: Please specify a model type to avoid inconsistent results due to incorrect prompt formatting.')
        sys.exit(1)


def dump_predictions(prompts, model_responses, path, model_identifier):
    dump = []
    dataset = model_identifier.split('_')[0]
    for prompt, model_response in zip(prompts, model_responses):
        dump.append({"instruction": prompt, "output": model_response, "generator": model_identifier, "dataset":dataset, "datasplit":'test'})
    
    with open(path + '.json', 'w') as file:
        json.dump(dump, file, indent=4) 
    logging.info('Logged inference results.')

def inference(test_inputs, model, sampling_params):
    with tqdm.tqdm(torch.no_grad()):
        intermediate_outputs =  model.generate(test_inputs, sampling_params)
        predictions = [intermediate_output.outputs[0].text for intermediate_output in intermediate_outputs]
    return predictions 

def load_model(args):    
    model = LLM(args.model_path)
    logging.info('Loaded model with VLLM ...')
    sampling_params = SamplingParams(max_tokens = 1024, logprobs=None)
    return model, sampling_params


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='/data/user_data/hdiddee/llama_models/llama_checkpoint/', help='path to hf model (or hf hub identifier will also work). Eg: meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument("--model_type", type=str, help='Required to format the prompts in accordance with the model class.')
    parser.add_argument("--identifier", type=str, help='Any file identifier you need to use for the file where the inferences are stored. ')
    parser.add_argument("--output_path", type=str, default = '../vllm_inferences')
    parser.add_argument("--max_samples", type = int, default = None)
    parser.add_argument("--benchmark_path", type=str, default = './curated_benchmark.csv', help="CSV of the prompts files. Also accepts hf_datasets.")

    args = parser.parse_args()
    
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
        logging.info('Making folder for storing the results ...')
    
    if not args.identifier:         
        logging.warning('Warning: Identifier not passed. Will be creating a default identifier;')
        model_identifier = args.model_path.split('/')[-1]
        logging.info(f'Model Identifier:{model_identifier}')
    
    
    model, sampling_params = load_model(args)
    prompts = pd.read_csv(args.benchmark_path, names=['Category', 'Prompt'])['Prompt'].to_list()[1:]
    test_inputs = model_wise_formatter(prompts, args.model_type)
    outputs = inference(test_inputs, model, sampling_params)
    
    dump_predictions(prompts, outputs, os.path.join(args.output_path, args.identifier), args.model_path)


