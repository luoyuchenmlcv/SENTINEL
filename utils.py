import torch
import torch.nn.functional as F
import ot  # POT library for Earth Mover's Distance
import numpy as np
from itertools import groupby
from operator import itemgetter
from sklearn.metrics import precision_recall_curve, auc, roc_curve, roc_auc_score

def generate_context_windows(sequence_length, window_sizes):
    
    if sequence_length<=window_sizes[0]:
        return torch.ones((2,sequence_length)).bool()
    masks = []
    positions = torch.arange(sequence_length).unsqueeze(0)  # Shape (1, sequence_length)
    for window_size in window_sizes:
        num_windows = sequence_length - window_size + 1
        window_starts = torch.arange(num_windows).unsqueeze(1)  # Shape (num_windows, 1)
        mask = (positions >= window_starts) & (positions < (window_starts + window_size))
        masks.append(mask)
    all_masks = torch.cat(masks, dim=0)
    return all_masks


def compute_token_weights(masks, window_weights):
  
    masks_float = masks.float()  # (num_masks, sequence_length)

    mask_counts = masks_float.sum(dim=1, keepdim=True)  # (num_masks, 1)

    distribution = (masks_float / mask_counts) * window_weights  # (num_masks, sequence_length)

    token_weights = distribution.sum(dim=0)  # (sequence_length,)

    token_exposure_counts = masks_float.sum(dim=0)  # (sequence_length,)

    token_weights_normalized = token_weights / token_exposure_counts

    return token_weights_normalized


def compute_windowed_features(masks, features):

    weights = masks.float()
    mask_counts = weights.sum(dim=1, keepdim=True) 
    weights = weights / mask_counts
    windowed_features = weights @ features 
    return windowed_features



def compute_loss(D_AB, D_AA, D_BB, mu_aug, nu_aug, alpha=0.3,  use_approx=False):
 
    mu_aug_probs = F.softmax(mu_aug, dim=0) 
    nu_aug_probs = F.softmax(nu_aug, dim=0) 
    

    if not use_approx:
        emd_loss = ot.emd2(mu_aug_probs.squeeze(), nu_aug_probs.squeeze(), D_AB, processes=1, log=False)
    else:
        emd_loss = mu_aug_probs.T @ D_AB @ nu_aug_probs

    
    intra_A = mu_aug_probs.T @ D_AA @ mu_aug_probs
    intra_B = nu_aug_probs.T @ D_BB @ nu_aug_probs

    
    loss = emd_loss - alpha*(intra_A + intra_B)
    return loss, [emd_loss.item(), intra_A.item(), intra_B.item()]



def tokens_to_words(token_list):
    """
    将transformer分词器的token list转换回word list
    
    Args:
        token_list: transformer分词器生成的token列表
    
    Returns:
        重构后的单词列表
    """
    words = []
    current_word = ""
    
    # 处理第一个token（可能没有Ġ前缀）
    if token_list and not str(token_list[0]).startswith('Ġ'):
        current_word = token_list[0]
    
    for token in token_list:
        # 如果token是字符串表示，去掉引号
        if isinstance(token, str) and token.startswith("'") and token.endswith("'"):
            token = token[1:-1]
        
        # 跳过纯换行符token
        if token in ['Ċ', 'ĊĊ']:
            continue
        
        # 移除token中的换行符
        token = token.replace('Ċ', '')
        
        # 如果清理后token为空，跳过
        if not token:
            continue
        
        # 如果token以Ġ开头，表示一个新单词
        if token.startswith('Ġ'):
            # 如果有当前单词，添加到列表
            if current_word:
                words.append(current_word)
            
            # 开始一个新单词，去掉Ġ前缀
            current_word = token[1:]
        else:
            # 当前单词的延续
            current_word += token
    
    # 添加最后一个单词（如果存在）
    if current_word:
        words.append(current_word)
    
    return words



def get_word_scores(tokens, scores, model_name):
    if 'vicuna' in model_name or 'llama2' in model_name or 'mistral' in model_name:
        
        words = []
        word_scores = []
        puncts = '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'
        for idx, (token, score) in enumerate(zip(tokens, scores)):
            
            if token.startswith('▁') and token not in puncts:
                compose_word = token[1:]
                word_score = [score]
                curr_pos = idx+1
                while (curr_pos < len(tokens)) and not (tokens[curr_pos].startswith('▁')) and (tokens[curr_pos] not in puncts): 
                    compose_word += tokens[curr_pos]
                    word_score.append(scores[curr_pos])
                    curr_pos +=1
                words.append(compose_word)
                word_score = sum(word_score)/len(word_score)
                word_scores.append(word_score)
            elif  token in puncts:
                words.append(token) 
                word_scores.append(score)
            else:
                continue

    elif 'llama_8b_3_1' or "qwen_14b_2_5" in model_name:
        
        words = []
        word_scores = []
    
        current_word = ""
        current_sum = 0.0     # 用于累加当前单词的 token 分数
        current_count = 0     # 当前单词包含的 token 数
    
        for token, score in zip(tokens, scores):
            # 1) 先按 “Ċ” 分割，因为“Ċ”表示换行/断开
            parts = token.split("Ċ")
    
            for part in parts:
                # 如果分割后得到的是空串，说明出现了 "ĊĊ" 或 token 末尾带有 "Ċ"
                # 这时相当于一个明确的“换行 / 新单词边界”
                if part == "":
                    # 若当前有正在累积的单词，需要将其封入结果
                    if current_word:
                        words.append(current_word)
                        word_scores.append(current_sum / current_count)
                        current_word = ""
                        current_sum = 0.0
                        current_count = 0
                    continue  # 直接处理下一个 part
    
                # 2) 去除开头所有的“Ġ”，并统计出现次数
                boundary_count = 0
                while part.startswith("Ġ"):
                    part = part[1:]
                    boundary_count += 1
    
                # 如果 boundary_count > 0，说明这里是一个新单词起点
                if boundary_count > 0:
                    # 如果当前有正在累积的单词，先把它存起来并结算分数
                    if current_word:
                        words.append(current_word)
                        word_scores.append(current_sum / current_count)
                        current_word = ""
                        current_sum = 0.0
                        current_count = 0
                    # 开始一个新单词
                    current_word = part
                    current_sum = score
                    current_count = 1
                else:
                    # 否则说明这是与前面 token 同一个单词的续写部分
                    current_word += part
                    current_sum += score
                    current_count += 1
    
        # 循环结束后，若还有未入列的单词，需要收尾
        if current_word:
            words.append(current_word)
            word_scores.append(current_sum / current_count)
            
    elif 'qwen' in model_name:
        words = []
        word_scores = []
        for idx, (token, score) in enumerate(zip(tokens, scores)):
            words.append(str(token)[2:-1])
            word_scores.append(score)
    words = np.array(words)
    word_scores = np.array(word_scores)

    # print("words", words)
    # print("word_scores", word_scores)
    word_scores = (word_scores-word_scores.min())/(word_scores.max()-word_scores.min()+1e-6)
    return words, word_scores







def extract_context_windows_token(tokenizer, token_ids, prompt_idx_start, prompt_idx_end, importance_scores, k1):
    # print("K1", k1)

    importance_scores = np.array(importance_scores)
    # print("importance_scores", importance_scores)
    significant_indices = np.where(importance_scores > k1)[0]
    # print("significant_indices", significant_indices)
    groups = []
    for k, g in groupby(enumerate(significant_indices), lambda ix: ix[0] - ix[1]):
        group = list(map(itemgetter(1), g))
        groups.append(group)
        
    # print("groups", groups)
    
    context_windows = []
    for group in groups:
        start = group[0]
        end = group[-1]
        # print("start", start)
        # print("end", end)
        window_token_ids = token_ids[:, max(prompt_idx_start, start):min(end+1, prompt_idx_end+1)]
        # print("window_token_ids", window_token_ids)

        # if window_token_ids.shape[1] >= 5:
        if window_token_ids.shape[1] >= 10000:
            # print("cw:",    tokenizer.batch_decode(window_token_ids))
            window_string = tokenizer.batch_decode(window_token_ids)[0]
            context_windows.append(window_string)
    context_windows = list(set(context_windows))
    
    return context_windows

    

def extract_context_windows(words, importance_scores, k1):
   

    importance_scores = np.array(importance_scores)

    significant_indices = np.where(importance_scores > k1)[0]

    groups = []
    for k, g in groupby(enumerate(significant_indices), lambda ix: ix[0] - ix[1]):
        group = list(map(itemgetter(1), g))
        groups.append(group)

    context_windows = []
    for group in groups:
        start = group[0]
        end = group[-1]
        window_words = words[start:end+1]
        if len(window_words) >= 5:
            window_string = ' '.join(window_words)
            context_windows.append(window_string)
    context_windows = list(set(context_windows))
    
    return context_windows



def extract_multiple_context_windows(words, importance_scores, k1_list):
    
    extracted_words = []
    for k1 in k1_list:
        # print("k1",k1)
        # print("importance_scores", importance_scores)
        k_extracted = extract_context_windows(words, importance_scores, k1)
        # print("k_extracted", k_extracted)
        extracted_words += k_extracted
    return list(set(extracted_words))

def extract_multiple_context_windows_token(tokenizer, token_ids, prompt_idx_start, prompt_idx_end, importance_scores, k1_list):
    
    extracted_words = []
    for k1 in k1_list:
        extracted_words += extract_context_windows_token(tokenizer, token_ids, prompt_idx_start, prompt_idx_end, importance_scores, k1)
    
    return list(set(extracted_words))
    
def evaluate(labels, scores):

    precision, recall, _ = precision_recall_curve(labels, scores)
    aupr_score = auc(recall, precision)
    fpr, tpr, _ = roc_curve(labels, scores)
    auroc_score = auc(fpr, tpr)
    return aupr_score, auroc_score


def compute_tpr_fpr(pred, label):
    """
    计算 True Positive Rate (TPR) 和 False Positive Rate (FPR)。
    
    参数:
    pred : 1D 数组或列表，表示模型预测的二分类标签 (0 或 1)
    label: 1D 数组或列表，表示真实标签 (0 或 1)
    
    返回:
    tpr, fpr: 浮点数
    """
    # 转成 numpy 数组，保证可以做布尔运算
    pred = np.array(pred)
    label = np.array(label)
    
    # 计算混淆矩阵
    # TP: 预测为 1，实际为 1
    # FN: 预测为 0，实际为 1
    # FP: 预测为 1，实际为 0
    # TN: 预测为 0，实际为 0
    TP = np.sum((pred == 1) & (label == 1))
    FN = np.sum((pred == 0) & (label == 1))
    FP = np.sum((pred == 1) & (label == 0))
    TN = np.sum((pred == 0) & (label == 0))
    
    # 计算 TPR (also called Recall or Sensitivity)
    # TPR = TP / (TP + FN)
    tpr = TP / (TP + FN) if (TP + FN) != 0 else 0.0
    
    # 计算 FPR
    # FPR = FP / (FP + TN)
    fpr = FP / (FP + TN) if (FP + TN) != 0 else 0.0
    print("FP", FP)
    return tpr, fpr

from PIL import Image, ImageDraw, ImageFont
import colorsys


def visualize_word_scores(tokens, scores, output_file, model_name):

    # if 'vicuna' in model_name or 'llama' in model_name or 'mistral' in model_name:
        
    # # 步骤1: 将token合并成单词，并计算每个单词的平均分数
    #     words = []
    #     word_scores = []
    #     puncts = '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'
    #     for idx, (token, score) in enumerate(zip(tokens, scores)):
            
    #         if token.startswith('▁') and token not in puncts:
    #             compose_word = token[1:]
    #             word_score = [score]
    #             curr_pos = idx+1
    #             while (curr_pos < len(tokens)) and not (tokens[curr_pos].startswith('▁')) and (tokens[curr_pos] not in puncts): 
    #                 compose_word += tokens[curr_pos]
    #                 word_score.append(scores[curr_pos])
    #                 curr_pos +=1
    #             words.append(compose_word)
    #             word_score = sum(word_score)/len(word_score)
    #             word_scores.append(word_score)
    #         elif  token in puncts:
    #             words.append(token) 
    #             word_scores.append(score)
    #         else:
    #             continue
                
    # if 'qwen' in model_name:
    #     words = []
    #     word_scores = []
    #     for idx, (token, score) in enumerate(zip(tokens, scores)):
    #         words.append(str(token)[2:-1])
    #         word_scores.append(score)

    # word_scores = np.array(word_scores)
    # word_scores = (word_scores-word_scores.min())/(word_scores.max()-word_scores.min()+1e-6)

    words, word_scores = get_word_scores(tokens, scores, model_name)
    
    # 步骤2: 计算颜色    
    def score_to_color(score):
        hue = 240 / 360  # 蓝色
        saturation = 0.7
        value = 0.9
        r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(hue, saturation, value)]
        alpha = int(score * 255)  # 透明度从0（完全透明）到255（不透明）
        return (r, g, b, alpha)
    
    # 步骤3: 创建图像（使用RGBA模式以支持透明度）
    word_font_size = 24
    score_font_size = 16
    word_font = ImageFont.truetype("DejaVuSerif-Bold.ttf", word_font_size)
    score_font = ImageFont.truetype("DejaVuSerif-Bold.ttf", score_font_size)
    
    padding = 10
    line_height = word_font_size + score_font_size + 12  # 调整行高，容纳分数
    word_spacing = 5
    max_words_per_line = 20

    # 计算每行的单词和总行数
    lines = []
    current_line = []
    current_line_width = 0
    max_line_width = 0
    for idx, word in enumerate(words):
        word_width = word_font.getbbox(word)[2]
        score_width = score_font.getbbox(f"{word_scores[idx]:.2f}")[2]
        total_width = max(word_width, score_width)
        if len(current_line) >= max_words_per_line or current_line_width + total_width > 1000:  # 假设最大宽度为1000像素
            lines.append(current_line)
            max_line_width = max(max_line_width, current_line_width)
            current_line = []
            current_line_width = 0
        current_line.append((word, word_scores[idx]))
        current_line_width += total_width + word_spacing
    if current_line:
        lines.append(current_line)
        max_line_width = max(max_line_width, current_line_width)
    
    # 计算图像大小
    img_width = int(max_line_width + padding * 2)
    img_height = int(len(lines) * line_height + padding * 2)
    
    img = Image.new('RGBA', (img_width, img_height), color=(255, 255, 255, 0))  # 透明背景
    draw = ImageDraw.Draw(img)
    
    # 步骤4: 在图像上绘制带透明背景的多行文本，并在每个单词上方显示分数
    y = padding
    for line in lines:
        x = padding
        for word, score in line:
            color = score_to_color(score)
            word_width = word_font.getbbox(word)[2]
            word_height = word_font.getbbox(word)[3]
            score_text = f"{score:.2f}"
            score_width = score_font.getbbox(score_text)[2]
            total_width = max(word_width, score_width)
            
            # 绘制半透明背景
            background = Image.new('RGBA', (int(total_width), line_height), color)
            img.paste(background, (int(x), int(y)), background)
            
            # 绘制分数（居中对齐）
            score_x = x + (total_width - score_width) / 2
            draw.text((score_x, y), score_text, font=score_font, fill=(0, 0, 0, 255))
            
            # 绘制单词（居中对齐）
            word_x = x + (total_width - word_width) / 2
            draw.text((word_x, y + score_font_size + 4), word, font=word_font, fill=(0, 0, 0, 255))
            
            x += total_width + word_spacing

        y += line_height
    
    # 保存图像
    img.save(output_file, format='png')
    print(f"带透明背景色的文本图像已保存到 {output_file}")
    return np.array(words), np.array(word_scores)


def concatenate_images(image_paths, output_path, direction='horizontal'):
    """
    拼接多张图像为一张图像。
    
    参数:
        image_paths (list): 图像文件路径列表
        output_path (str): 拼接后的输出图像路径
        direction (str): 拼接方向, 'horizontal' 为横向, 'vertical' 为纵向
    """
    # 打开所有图像
    images = [Image.open(img) for img in image_paths]

    # 获取每张图像的宽度和高度
    widths, heights = zip(*(img.size for img in images))

    if direction == 'horizontal':
        # 横向拼接时，总宽度是所有图像宽度之和，高度为最高的图像高度
        total_width = sum(widths)
        max_height = max(heights)
        # 创建一个新的空白图像
        concatenated_image = Image.new('RGB', (total_width, max_height), 'white')

        # 将图像逐个粘贴到新图像中
        x_offset = 0
        for img in images:
            concatenated_image.paste(img, (x_offset, 0))
            x_offset += img.width

    elif direction == 'vertical':
        # 纵向拼接时，总高度是所有图像高度之和，宽度为最宽的图像宽度
        total_height = sum(heights)
        max_width = max(widths)
        # 创建一个新的空白图像
        concatenated_image = Image.new('RGB', (max_width, total_height))

        # 将图像逐个粘贴到新图像中
        y_offset = 0
        for img in images:
            concatenated_image.paste(img, (0, y_offset))
            y_offset += img.height

    else:
        raise ValueError("拼接方向仅支持 'horizontal' 或 'vertical'")

    # 保存拼接后的图像
    concatenated_image.save(output_path)
    print(f"图像已成功拼接并保存到 {output_path}")











import torch
from functools import lru_cache


def build_ascii_mask(tokenizer, final_vocab_size) -> torch.BoolTensor:
    """
    生成 (final_vocab_size,) 的 bool 掩码。
    • 若 id < len(tokenizer) ⇒ 用 tokenizer.decode 判断 .isascii()
    • 其余 id 统一视为非 ASCII（False）
    """
    if final_vocab_size is None:
        # 让用户传 logits.shape[-1] 可以最稳妥
        raise ValueError("请显式传入最终 vocab_size（通常 = logits.shape[-1]）")

    base_len = len(tokenizer)
    mask = [tokenizer.decode([i], skip_special_tokens=False).isascii() for i in range(base_len)]
    if final_vocab_size > base_len:                         # 对齐剩余列
        mask.extend([False] * (final_vocab_size - base_len))
    return torch.tensor(mask, dtype=torch.bool, device='cuda:0')


    
def suppress_non_ascii_logits(
    logits: torch.Tensor,
    ascii_mask: torch.BoolTensor,
    min_value: float = -100,
) -> torch.Tensor:
    """
    将 logits 中所有非 ASCII token 的值置为 min_value 并返回。

    参数
    ----
    logits: (..., vocab_size) 的 Tensor
    ascii_mask: (vocab_size,) 的 bool Tensor，True 表示 ASCII
    min_value: 替换值，默认 -1e9
    """
    mask = ascii_mask.to(logits.device)
    logits = logits.clone()          # 纯函数，不修改原 tensor
    logits[..., ~mask] = min_value   # 抑制非 ASCII
    return logits




def compute_seq_prob(model, tokenizer, prompt, target, batch_size=5):

    prompt = np.array(prompt)
    target = np.array(target)
    

    import itertools  
    concat_inputs = np.array(list(itertools.product(prompt, target)))
    concat_indices = list(itertools.product(range(len(prompt)), range(len(target))))
    perplexities = []
    lls = []
    ranks = []
    with torch.no_grad():
    
        batches = np.arange(len(concat_inputs)//batch_size*batch_size).reshape(-1, batch_size).tolist()
        batches.append([len(concat_inputs)//batch_size*batch_size+i for i in range(len(concat_inputs)%batch_size)])
        batches = [b for b in batches if b]
        for batch_idx in batches:
            
            batch_concat_inputs = [i+j for i,j in concat_inputs[batch_idx]]
            batch_inputs = tokenizer(batch_concat_inputs, return_tensors = "pt", padding = True)
            batch_target = [j for i,j in concat_inputs[batch_idx]]

            labels = torch.zeros(batch_inputs.input_ids.shape, device='cuda:0', dtype=torch.long)-100
            batch_target_len = [len(i) for i in tokenizer(batch_target, add_special_tokens=False).input_ids]  #some tokenizer may add bos token here
    
            for i, (t, t_len) in enumerate(zip(batch_target, batch_target_len)):
                labels[i, -t_len:] = batch_inputs.input_ids[i, -t_len:]
            batch_inputs['labels'] = labels


            
            # print("batch_inputs", self.tokenizer.batch_decode(batch_inputs.input_ids))
            # print("batch_labels", self.tokenizer.batch_decode(labels))
            generation_outputs = model(**batch_inputs.to('cuda:0'))
            logits = generation_outputs.logits
            
            from torch.nn import CrossEntropyLoss 
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # print("shift_logits", shift_logits.shape)
            # print("shift_labels", shift_labels)
            # print("shift_labels!=-100", (shift_labels[0]!=-100).nonzero(), (shift_labels[0]!=-100).nonzero().squeeze())
            # print("shift_logits pred:", shift_logits[:,(shift_labels[0]!=-100).nonzero().squeeze(),:].shape, shift_logits[:,(shift_labels[0]!=-100).nonzero().squeeze(),:].argmax(-1))
            # print("shift_logits pred:", shift_logits[:,(shift_labels[0]!=-100).nonzero()[0]])

    
            shift_logits_rank = (-shift_logits).argsort(-1).argsort(-1)

            
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction='none')
            shift_logits = shift_logits.view(-1, logits.shape[-1]) #b, seq_len, vocab_size
            
            
            
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            losses = loss_fct(shift_logits, shift_labels)
            losses = losses.view(batch_inputs.input_ids.shape[0], -1)
            # prob_token_wise = [torch.exp(-row[row != 0]) for row in losses]
            # print("losses", losses)
            # print("element-wise rank:", [row_rank[row_loss != 0] for row_loss, row_rank in zip(losses, shift_logits_rank)] )
            for row_loss, row_label, row_rank in zip(losses, labels, shift_logits_rank):
                # print("row_loss", row_loss)
                # print("row label", row_label[row_label!=-100])
                row_label = row_label[row_label!=-100]
                # print("row_rank", row_rank[row_loss != 0, row_label])
                ranks.append(row_rank[row_loss != 0, row_label].tolist())

                
            batch_nlls = losses.sum(dim=1)  
            lls.append(-batch_nlls.cpu().numpy())
            
        lls = np.hstack(lls)

        ranks_matrix  = np.empty((len(prompt), len(target)), dtype=object)
        ranks_matrix[list(zip(*concat_indices))[0], list(zip(*concat_indices))[1]] = ranks

        lls_matrix = np.zeros((len(prompt), len(target)))
        lls_matrix[list(zip(*concat_indices))[0], list(zip(*concat_indices))[1]] = lls.copy()
        

    return lls_matrix, ranks_matrix
