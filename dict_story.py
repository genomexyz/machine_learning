import numpy as np
import random

#setting
sampel_train = 'sampel_mix.txt'
len_input_char = 80
len_generated_story = 1500

def predict_char(d, fix_param):
    try:
        total_mem = []
        variation = []
        pred_candidate = d[fix_param]
        for iter_candidate in range(len(pred_candidate)):
            single_candidate = pred_candidate[iter_candidate]
            try:
                idx_var = variation.index(single_candidate)
                total_mem[idx_var] += 1
            except ValueError:
                variation.append(single_candidate)
                total_mem.append(1)
        idx_chosen = np.argmax(total_mem)
        return variation[idx_chosen]
    except KeyError:
        return random.choice(char_set)

sampel_open = open(sampel_train)
sampel_raw = sampel_open.read()

char_set = sorted(list(set(sampel_raw)))

mega_dict = {}
longest_ops = 1
for i in range(len(sampel_raw)-len_input_char-1):
    code = sampel_raw[i:i+len_input_char]
    try:
        mega_dict[code].append(sampel_raw[i+len_input_char])
        if len(mega_dict[code]) > longest_ops:
            longest_ops = len(mega_dict[code])
    except KeyError:
        mega_dict[code] = [sampel_raw[i+len_input_char]]

#generate story
trigger_sentence = 'this is trigger'
#trigger_sentence = 'Loki: [Giving up on fighting against Thanos] You will... never be... a god. [Tha'
trigger_sentence = '[At the cells where prisoners are lead into. One of the guards knocks the hat fr'
generated_story = trigger_sentence
for i in range(len_generated_story):
    if len(trigger_sentence) < len_input_char:
        dif = len_input_char - len(trigger_sentence)
        fix_param = ' '*dif+trigger_sentence
    elif len(trigger_sentence) > len_input_char:
        fix_param = trigger_sentence[-len_input_char:]
    else:
        fix_param = trigger_sentence
    pred_char = predict_char(mega_dict, fix_param)
    
    generated_story += pred_char
    trigger_sentence = fix_param[1:]
    trigger_sentence += pred_char

print('below is generated story:')
print(generated_story)