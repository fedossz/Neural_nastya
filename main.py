#!/usr/bin/env python3
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
"""




"""Just copied generate_transformers.py and changed it a bot so it looks fine

first need to copy model repository via git clone https://github.com/sberbank-ai/ru-gpts

Then go to ru-gpts folder and install all required packages

pip install -r ru-gpts/requirements.txt
(better use virtualenv)
"""
import os

import argparse
import logging

import random
import time

import numpy as np
import torch

import sqlite3

conn = sqlite3.connect('user_data.db', check_same_thread = False)
cursor = conn.cursor()

cursor.execute('CREATE TABLE IF NOT EXISTS user_data (id INTEGER, chat TEXT, last_context TEXT)')

from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
)

#needs to be installed via pip3 install pytelegrambotapi
import telebot

bot = telebot.TeleBot('token')

bot.send_message(your_id, 'Я готова')



logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer),
}

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PADDING_TEXT = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


#
# Functions to prepare models' input
#


def prepare_ctrl_input(args, _, tokenizer, prompt_text):
    if args.temperature > 0.7:
        logger.info("CTRL typically works better with lower temperatures (and lower top_k).")

    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
    if not any(encoded_prompt[0] == x for x in tokenizer.control_codes.values()):
        logger.info("WARNING! You are not starting your generation from a control code so you won't get good results")
    return prompt_text


def prepare_xlm_input(args, model, tokenizer, prompt_text):
    # kwargs = {"language": None, "mask_token_id": None}

    # Set the language
    use_lang_emb = hasattr(model.config, "use_lang_emb") and model.config.use_lang_emb
    if hasattr(model.config, "lang2id") and use_lang_emb:
        available_languages = model.config.lang2id.keys()
        if args.xlm_language in available_languages:
            language = args.xlm_language
        else:
            language = None
            while language not in available_languages:
                language = input("Using XLM. Select language in " + str(list(available_languages)) + " >>> ")

        model.config.lang_id = model.config.lang2id[language]
        # kwargs["language"] = tokenizer.lang2id[language]

    # TODO fix mask_token_id setup when configurations will be synchronized between models and tokenizers
    # XLM masked-language modeling (MLM) models need masked token
    # is_xlm_mlm = "mlm" in args.model_name_or_path
    # if is_xlm_mlm:
    #     kwargs["mask_token_id"] = tokenizer.mask_token_id

    return prompt_text


def prepare_xlnet_input(args, _, tokenizer, prompt_text):
    prompt_text = (args.padding_text if args.padding_text else PADDING_TEXT) + prompt_text
    return prompt_text


def prepare_transfoxl_input(args, _, tokenizer, prompt_text):
    prompt_text = (args.padding_text if args.padding_text else PADDING_TEXT) + prompt_text
    return prompt_text


PREPROCESSING_FUNCTIONS = {
    "ctrl": prepare_ctrl_input,
    "xlm": prepare_xlm_input,
    "xlnet": prepare_xlnet_input,
    "transfo-xl": prepare_transfoxl_input,
}


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default='gpt2',
        type=str,
        required=False,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default='sberbank-ai/rugpt3large_based_on_gpt2',#input('model name or path? - '),
        type=str,
        required=False,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )

    parser.add_argument("--prompt", type=str, default="")
    #parser.add_argument("--length", type=int, default=98)
    parser.add_argument("--stop_token", type=str, default="</s>", help="Token at which text generation is stopped")

    """
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(input('temperaure - ')),
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    """
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
#    parser.add_argument("--k", type=int, default=int(input('top_k - '))) # now testing temp 1.01 top-k 26 and p 0.93
#    parser.add_argument("--p", type=float, default=float(input('p - ')))

    parser.add_argument("--padding_text", type=str, default="", help="Padding text for Transfo-XL and XLNet.")
    parser.add_argument("--xlm_language", type=str, default="", help="Optional language when used with the XLM model.")

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    #args = parser.parse_args()
    args, unknown = parser.parse_known_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    set_seed(args)
    # Initialize the model and tokenizer
    try:
        args.model_type = args.model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    except KeyError:
        raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    model.to(args.device)

    #args.length = adjust_length_to_model(args.length, max_sequence_length=model.config.max_position_embeddings)
    logger.info(args)
    generated_sequences = []
    delete_hist_replies = ['Всё, теперь я о вас ничего не помню!', 'Хорошо, я всё забыла', 'Все ваши данные удалены', 'Теперь я не знаю кто вы такой', 'Хорошо, давай начнём все заново']
    bot.send_message(your_id, 'Терь все, ошибок нет можно начинать!')# message that informs you that all the weights loaded and bot ready to work, cause weights loading can take from 2 minutes and more
    #declaring it as global cause error appears if i won't do that
    global gen_temp
    global gen_k
    global gen_p
    global gen_length
    #getting text generation properties from user input
    #if u don't know what they mean i left best ones
    gen_temp = float(input('temperature - ')) #1.01 how "random" will be the result
    gen_k = int(input('top_k - ')) #26
    gen_p = float(input('top_p - ')) #0.93
    gen_length = int(input('gen_length - ')) #length of generated text, i use values between 54-92
    @bot.message_handler(content_types=['text'])
    def sned_resp(message):
        global gen_temp
        global gen_k
        global gen_p
        global gen_length
        tipe_per = 'def'
        if message.text.lower()[:3].replace(' ', '') in ['|v|', '/f', '/d']:
          start_cmd_key = message.text.lower()[:3]
          text_to_start_gen = message.text.replace(start_cmd_key, '')
        elif message.text.lower() == '/start':
          bot.send_message(message.chat.id, 'Начните разговор просто написав приветствие или что-либо другое, на него вам ответит нейросеть, ответ может занять время.\nНейросеть зовут Анастасия, чтобы она «забыла» весь ваш разговор отправьте /drop или /clear')
          return
        elif message.text.lower() in ['/drop', '/clean', '/clear', '/forget']:
          cursor.execute(f'DELETE from user_data WHERE id = {message.chat.id}')
          conn.commit()
          bot.send_message(message.chat.id, random.choice(delete_hist_replies))
          return
        elif (message.text.lower().startswith('/adj')) & (message.chat.id == your_id):
          #gen_temp, gen_k, gen_p, gen_length = message.text.lower().split()[-4:]
          gen_temp = float(message.text.lower().split()[1])
          gen_k = int(message.text.lower().split()[2])
          gen_p = float(message.text.lower().split()[3])
          gen_length = int(message.text.lower().split()[4])
          bot.send_message(message.chat.id, 'Значения генерации текста изменены, со следующего сообщения всё изменится (я не обещаю)')
          return
        else:
          tipe_per = 'dial'
          cursor.execute('SELECT * FROM user_data WHERE id = ?', (message.chat.id, ))
          if cursor.fetchall():
            cursor.execute('SELECT last_context FROM user_data WHERE id = ?', (message.chat.id, ))
            last_context_str_td = cursor.fetchall()[0]
          else:
            new_user_dat = [int(message.chat.id), str(message.chat), '']
            cursor.executemany('INSERT INTO user_data VALUES(?, ?, ?)', (new_user_dat, ))
            last_context_str_td = ''
          print(last_context_str_td)
          #last_context_str_td = last_context_str_td.replace('|end-context-?|', '')
          ddf_s = ''
          for i in range(len(last_context_str_td)):
            ddf_s += '{} '
          last_context_str_td = ddf_s.format(*last_context_str_td)
          #print(f'LAST CONTEXT[[[[[[[[[[{last_context_str_td}]]]]]]]]]')
          text_to_start_gen = f'''
{last_context_str_td.replace('|end-context-?|', '')}
Собеседник: {message.text}
Настя: '''
        print(text_to_start_gen)
        conn.commit()
        bot.send_chat_action(message.chat.id, 'typing')
        prompt_text = ""
        while not len(prompt_text):
            prompt_text = args.prompt if args.prompt else text_to_start_gen#input("Context >>> ")

        # Different models need different input formatting and/or extra arguments
        requires_preprocessing = args.model_type in PREPROCESSING_FUNCTIONS.keys()
        if requires_preprocessing:
            prepare_input = PREPROCESSING_FUNCTIONS.get(args.model_type)
            preprocessed_prompt_text = prepare_input(args, model, tokenizer, prompt_text)
            encoded_prompt = tokenizer.encode(
                preprocessed_prompt_text, add_special_tokens=False, return_tensors="pt", add_space_before_punct_symbol=True
            )
        else:
            encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
        encoded_prompt = encoded_prompt.to(args.device)




        gen_length = adjust_length_to_model(gen_length, max_sequence_length=model.config.max_position_embeddings)
        output_sequences = model.generate(
            input_ids=encoded_prompt,
            max_length=gen_length + len(encoded_prompt[0]),
            temperature=gen_temp,#args.temperature,
            top_k=gen_k,#args.k,
            top_p=gen_p,#args.p,
            repetition_penalty=args.repetition_penalty,
            do_sample=True,
            num_return_sequences=args.num_return_sequences,
        )

        # Remove the batch dimension when returning multiple sequences
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
            print("ruGPT:".format(generated_sequence_idx + 1))
            generated_sequence = generated_sequence.tolist()

            # Decode text
            text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

            # Remove all text after the stop token
            text = text[: text.find(args.stop_token) if args.stop_token else None]

            # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
            total_sequence = (
                prompt_text + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
            )

            generated_sequences.append(total_sequence)
            # os.system('clear')
            print(total_sequence)
            #print(total_sequence.encode('utf-8'))
            if tipe_per == 'dial':
              #neuro_reply = total_sequence.encode('utf-8').split(b'\n\n')[0].decode('utf-8').split('Я: ')[1].split('Собеседник: ')[0].split('Он:')[0]
              neuro_reply = total_sequence.split('Настя: ')[len(last_context_str_td.split('|end-context-?|'))].split('Собеседник: ')[0].split('Он:')[0]
              text_frfs = f'''
{last_context_str_td}
Собеседник: {message.text}
Настя: '''
              new_last_context = text_frfs + neuro_reply + '|end-context-?|\n'
              if len(new_last_context.split('|end-context-?|')) > 20:
                print('[][][]LAST CONTEXT SPLIT[][][]')
                print(last_context_str_td.split('|end-context-?|'))
                n_new_last_context = ''
                for i in range(19):
                  n_new_last_context += new_last_context.split('|end-context-?|')[i + 1] + '|end-context-?|'
                new_last_context = n_new_last_context# + neuro_reply + '|end-context-?|\n'
              last_context_str_td = last_context_str_td.replace(' \n', '').replace('  \n', '').replace('\n\n', '').replace('\n \n', '')
              print(new_last_context)
              bot.send_chat_action(message.chat.id, 'typing')
              """
              if len(neuro_reply) > 15:
                for i in range(len(neuro_reply)):
                  time.sleep(0.007)
              """
              cursor.execute(f'UPDATE user_data SET last_context = ? WHERE id = {message.chat.id}', (new_last_context, ))
            else:
              neuro_reply = total_sequence
            
            bot.send_message(message.chat.id, neuro_reply)
            conn.commit()
        prompt_text = "stop"
        #if args.prompt:
            #break
    bot.polling()

    return generated_sequences


if __name__ == "__main__":
    main()
