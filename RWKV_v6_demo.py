########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
import types, torch
import torch.nn as nn
from torch.nn import functional as F

MyModule = torch.jit.ScriptModule
MyFunction = torch.jit.script_method

class RWKV_TOKENIZER():
    table: list[list[list[bytes]]]
    good: list[set[int]]
    wlen: list[int]
    def __init__(self, file_name):
        self.idx2token = {}
        sorted = [] # must be already sorted
        lines = open(file_name, "r", encoding="utf-8").readlines()
        for l in lines:
            idx = int(l[:l.index(' ')])
            x = eval(l[l.index(' '):l.rindex(' ')])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(' '):])
            sorted += [x]
            self.idx2token[idx] = x

        self.token2idx = {}
        for k, v in self.idx2token.items():
            self.token2idx[v] = int(k)

        # precompute some tables for fast matching
        self.table = [[[] for j in range(256)] for i in range(256)]
        self.good = [set() for i in range(256)]
        self.wlen = [0 for i in range(256)]

        for i in reversed(range(len(sorted))): # reverse order - match longer tokens first
            s = sorted[i]
            if len(s) >= 2:
                s0 = int(s[0])
                s1 = int(s[1])
                self.table[s0][s1] += [s]
                self.wlen[s0] = max(self.wlen[s0], len(s))
                self.good[s0].add(s1)

    def encodeBytes(self, src: bytes) -> list[int]:
        src_len: int = len(src)
        tokens: list[int] = []
        i: int = 0
        while i < src_len:
            s: bytes = src[i : i + 1]

            if i < src_len - 1:
                s1: int = int(src[i + 1])
                s0: int = int(src[i])
                if s1 in self.good[s0]:
                    sss: bytes = src[i : i + self.wlen[s0]]
                    try:
                        s = next(filter(sss.startswith, self.table[s0][s1]))
                    except:
                        pass
            tokens.append(self.token2idx[s])
            i += len(s)

        return tokens

    def decodeBytes(self, tokens):
        return b''.join(map(lambda i: self.idx2token[i], tokens))

    def encode(self, src: str):
        return self.encodeBytes(src.encode("utf-8"))

    def decode(self, tokens):
        return self.decodeBytes(tokens).decode('utf-8')

    def printTokens(self, tokens):
        for i in tokens:
            s = self.idx2token[i]
            try:
                s = s.decode('utf-8')
            except:
                pass
            print(f'{repr(s)}{i}', end=' ')
            # print(repr(s), i)
        print()

########################################################################################################

# 这段代码是一个用于生成随机样本的函数。
# 这是一个函数定义，函数名为 sample_logits，接受三个参数 out、temperature 
# 和 top_p，其中 temperature 默认值为 1.0，top_p 默认值为 0.8。
def sample_logits(out, temperature=1.0, top_p=0.8):
    # 这行代码使用 softmax 函数对 out 进行操作，将输出转换为概率分布。
    # dim=-1 表示在最后一个维度上进行 softmax 操作。.numpy() 将结果转换为 NumPy 数组。
    probs = F.softmax(out, dim=-1).numpy()
    
    # 这行代码使用 NumPy 的 np.sort 函数对概率分布进行排序，
    # 并通过 [::-1] 实现降序排列。结果保存在 sorted_probs 变量中。
    sorted_probs = np.sort(probs)[::-1]
    # 这行代码计算累积概率，使用 NumPy 的 np.cumsum 函数对 sorted_probs 
    # 进行累加操作。结果保存在 cumulative_probs 变量中。
    cumulative_probs = np.cumsum(sorted_probs)
    # 这行代码通过比较 cumulative_probs 是否大于 top_p 来找到概率分布中的截断点。
    # np.argmax 返回第一个满足条件的索引，float() 将其转换为浮点数并保存在 cutoff 变量中。
    cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
    # 这行代码将低于 cutoff 的概率值设为 0，即将概率分布中小于截断点的概率置零。
    probs[probs < cutoff] = 0
    # 这段代码根据 temperature 的取值对概率分布进行调整。
    # 如果 temperature 不等于 1.0，则将概率分布的每个元素取倒数的 1.0 / temperature 次幂。
    if temperature != 1.0:
        probs = probs.pow(1.0 / temperature)
    # 这行代码将概率分布归一化，确保所有概率的总和为 1。
    probs = probs / np.sum(probs)
    # 这行代码使用 np.random.choice 函数根据概率分布 probs 生成一个随机样本，
    # a=len(probs) 表示可选的样本范围为 probs 的长度，p=probs 表示每个样本被选中的概率。
    out = np.random.choice(a=len(probs), p=probs)
    # 函数返回生成的随机样本。
    return out

########################################################################################################
# 加载一个分词器
tokenizer = RWKV_TOKENIZER("tokenizer/rwkv_vocab_v20230424.txt")
# 使用types.SimpleNamespace()创建一个简单的命名空间对象args，并为其设置以下属性：
args = types.SimpleNamespace()
args.MODEL_NAME = '/mnt/e/RWKV-Runner/models/rwkv-final-v6-1b5'
args.n_layer = 24 # 模型的层数。
args.n_embd = 2048 # 模型的嵌入维度。
args.vocab_size = 65536 #词表大小

context = "\nElon Musk has"
# context = "\n我们发现"
NUM_TRIALS = 3# 尝试生成文本的次数。
LENGTH_PER_TRIAL = 100# 每次尝试生成的文本长度。
TEMPERATURE = 1.0 # 控制生成文本的随机性的参数。值越大，生成的文本越随机；值越小，生成的文本越确定。
TOP_P = 0.7# 在生成文本时，只考虑累积概率超过此值的词汇。

class RWKV_RNN(MyModule):
    def __init__(self, args):
        super().__init__()
        self.args = args  # 将传入的args参数赋值给类的属性args。
         # 将模型设置为评估模式，这意味着模型中的dropout和batchnorm将被禁用。
        self.eval() # set torch to inference mode
        # 从指定路径加载模型权重，并确保权重被加载到CPU上。
        w = torch.load(args.MODEL_NAME + '.pth', map_location='cpu')
        # 这几行代码对加载的权重进行了处理。它们检查权重的键名，并根据键名对权重进行不同的操作。
        for k in w.keys():
            w[k] = w[k].float() # convert to f32 type
            if      '.time_' in k: w[k] = w[k].squeeze()
            if '.time_faaaa' in k: w[k] = w[k].unsqueeze(-1)

        self.n_head = w['blocks.0.att.time_faaaa'].shape[0]
        self.head_size = w['blocks.0.ln1.weight'].shape[0] // self.n_head
         # 创建一个新的命名空间对象，并将其赋值给self.w。
        self.w = types.SimpleNamespace() # set self.w from w
        self.w.blocks = {}  # 在self.w中创建一个名为blocks的字典。
        # for k in w.keys(): - 遍历字典w的所有键。注释中的例子 
         # 说明了代码的目标：将点分隔的键转换为嵌套的属性访问。
        for k in w.keys(): # example: "blocks.0.att.time_first" => self.w.blocks[0].att.time_first
            parts = k.split('.') #  使用.作为分隔符将键k分割成多个部分，并将结果存储在parts列表中。
            last = parts.pop() # 从parts列表中弹出最后一个元素并存储在last中。这将是要设置的属性的名称。
            #  初始化一个变量here，它将用于遍历或创建self.w中的嵌套命名空间。
            here = self.w
             # 遍历parts列表中的每个部分。
            for p in parts:
                if p.isdigit():
                    p = int(p)
                    # 如果当前数字键p不在here中，则在here中为其创建一个新的命名空间。
                    if p not in here: here[p] = types.SimpleNamespace()
                    # 更新here以指向新创建的或已存在的命名空间。
                    here = here[p]
                else:
                    # 如果当前部分p不是数字。
                    if not hasattr(here, p): setattr(here, p, types.SimpleNamespace())
                    here = getattr(here, p)
            setattr(here, last, w[k])

    def layer_norm(self, x, w):
        return F.layer_norm(x, (self.args.n_embd,), weight=w.weight, bias=w.bias)

    @MyFunction
    def channel_mixing(self, x, state, i:int, time_maa_k, time_maa_r, kw, vw, rw):
        i0 = (2+self.head_size)*i+0 #获取前文channel-mixing的最后一个token特征的位置（state[i0]取出），i为对应层数
        sx = state[i0] - x #经典的token_shift 将相邻两个token的通道进行混合
        #使用可学习参数分别计算出xk，xr（time_maa控制sx权重）
        xk = x + sx * time_maa_k
        xr = x + sx * time_maa_r
        state[i0] = x #保存当前最后一个混合后的token，留待与下一个新token重新进行混合
        r = torch.sigmoid(rw @ xr) #sigmoid 归一化，r可以理解门控单元
        k = torch.square(torch.relu(kw @ xk)) # square relu, primer paper
        return r * (vw @ k) #kv运算并通过遗忘门r

    @MyFunction
    def time_mixing(self, x, state, i:int, x_maa, w_maa, k_maa, v_maa, r_maa, g_maa, tm_w1, tm_w2, td_w1, td_w2, time_first, time_decay, kw, vw, rw, gw, ow, ln_w, ln_b):
        H = self.n_head
        S = self.head_size

        i1 = (2+S)*i+1  #获取前文time-mixing的最后一个token特征的位置（state[i0]取出），i为对应层数
        sx = state[i1] - x #经典的token_shift 将相邻两个token的通道进行混合
        state[i1] = x #保存当前最后一个混合后的token，留待与下一个新token重新进行混合
        xxx = x + sx * x_maa #使用可学习参数计算（time_maa控制sx权重）
        # data-dependent 这里使用了两个lora矩阵tm_w1，tm_w2进行矩阵乘
        xxx = torch.tanh(xxx @ tm_w1).view(5, 1, -1)
        xxx = torch.bmm(xxx, tm_w2).view(5, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0) #lora矩阵中的5*对应wkvrg 5个参数，所以这里通过unbind分出各自的特征
        #使用可学习参数*并与x相加（time_maa控制sx权重）
        xw = x + sx * (w_maa + mw)
        xk = x + sx * (k_maa + mk)
        xv = x + sx * (v_maa + mv)
        xr = x + sx * (r_maa + mr)
        xg = x + sx * (g_maa + mg)

        w = (time_decay + (torch.tanh(xw @ td_w1) @ td_w2).float()).view(H, S, 1) # data-dependent-decay 根据当前token计算出不同位置的decay（rwkv5的w没有T维度，缺少data-dependent）
        w = torch.exp(-torch.exp(w.float())) #exp的存在，为了保证数值稳定性 最终w的范围会被控制在0-1

        #分别与各自的可学习参数进行计算得到rkvg，并进行view以便后续计算
        r = (rw @ xr).view(H, 1, S) #r类似于遗忘门
        k = (kw @ xk).view(H, S, 1)
        v = (vw @ xv).view(H, 1, S)
        g = F.silu(gw @ xg) #g类似于输出门

        s = state[(2+S)*i+2:(2+S)*(i+1), :].reshape(H, S, S)#获取对应位置的state并调整维度以便计算

        x = torch.zeros(H, S) #创建一个为0的tensor用于存储结果
        a = k @ v #计算kv，transformer系列需要保留kv cache，而rwkv只需要计算当前kv
        x = r @ (time_first * a + s) #经过r门控单元得到x输出（time_first * a 的time_first决定kv中重要的特征的去留）
        s = a + w * s #计算当前state，a为当前kv，s为历史state（w*s的w决定历史state中重要的特征的去留，与遗忘门原理类似）
    
        state[(2+S)*i+2:(2+S)*(i+1), :] = s.reshape(S, -1) #存储最后一个state信息，以便下一个token的计算
        x = x.flatten() #展平x，以便后续group_norm计算

        x = F.group_norm(x.unsqueeze(0), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).squeeze(0) * g # same as gn(x/8, eps=1e-5) 先经过group_norm再与输出门g计算得到x
        return ow @ x #最后再乘以一个ow可学习参数

        # 定义forward方法，它接受两个参数：token和state。
    def forward(self, token, state):
            # 这是一个上下文管理器，确保在此代码块中不会计算任何梯度。
            # 这通常用于评估模式，以提高性能并避免不必要的计算。
            with torch.no_grad():
                # 如果state为None，则初始化state为一个全零张量。
                # 其形状由self.args.n_layer和self.args.n_embd确定。
                if state == None:
                    state = torch.zeros(self.args.n_layer * (2+self.head_size), self.args.n_embd)
                # 使用token索引self.w.emb.weight，获取词嵌入向量。
                x = self.w.emb.weight[token]
                # 对获取的词嵌入向量x应用层归一化。
                x = self.layer_norm(x, self.w.blocks[0].ln0)
                # 遍历每一层
                for i in range(self.args.n_layer):
                    att = self.w.blocks[i].att # 获取当前层的注意力参数
                    # 这些行使用time_mixing方法对x进行处理，并将结果加到x上。
                    x = x + self.time_mixing(self.layer_norm(x, self.w.blocks[i].ln1), state, i,
                        att.time_maa_x, att.time_maa_w, att.time_maa_k, att.time_maa_v, att.time_maa_r, att.time_maa_g, att.time_maa_w1, att.time_maa_w2,
                        att.time_decay_w1, att.time_decay_w2, att.time_faaaa, att.time_decay,
                        att.key.weight, att.value.weight, att.receptance.weight, att.gate.weight, att.output.weight,
                        att.ln_x.weight, att.ln_x.bias)
                    ffn = self.w.blocks[i].ffn # 获取当前层的前馈网络参数。
                    # 使用channel_mixing方法对x进行处理，并将结果加到x上。
                    x = x + self.channel_mixing(self.layer_norm(x, self.w.blocks[i].ln2), state, i, 
                        ffn.time_maa_k, ffn.time_maa_r, 
                        ffn.key.weight, ffn.value.weight, ffn.receptance.weight)
                
                # 对x应用最后的层归一化，并与self.w.head.weight进行矩阵乘法。
                x = self.w.head.weight @ self.layer_norm(x, self.w.ln_out)
                return x.float(), state


# 打印使用 CPU 加载模型的信息，其中 args.MODEL_NAME 是模型名称。
print(f'\nUsing CPU. Loading {args.MODEL_NAME} ...')
# 创建一个名为 model 的 RWKV_RNN 模型实例，参数为 args。
model = RWKV_RNN(args)

# 打印预处理上下文信息的提示，提示使用的是较慢的版本。然后初始化 init_state 为 None。
print(f'\nPreprocessing context (slow version. see v2/rwkv/model.py for fast version)')
init_state = None
# 对上下文进行分词编码，并使用模型的 forward 方法逐个处理分词编码的 tokens，
# 将结果保存在 init_out 和 init_state 中。
for token in tokenizer.encode(context).ids:
    init_out, init_state = model.forward(token, init_state)

# 使用循环进行多次试验（NUM_TRIALS 次）。
for TRIAL in range(NUM_TRIALS):
    # 在每次试验的开始打印试验信息和上下文。创建一个空列表 all_tokens 用于保存生成的 tokens。
    print(f'\n\n--[ Trial {TRIAL} ]-----------------', context, end="")
    all_tokens = []
    # 初始化变量 out_last 为 0，out 和 state 分别为 init_out 和 init_state 的克隆。
    out_last = 0
    out, state = init_out.clone(), init_state.clone()
    # 在每个试验中，使用循环生成 LENGTH_PER_TRIAL 个 tokens。
    for i in range(LENGTH_PER_TRIAL):
        # 调用 sample_logits 函数生成一个随机 token，并将其添加到 all_tokens 列表中。
        token = sample_logits(out, TEMPERATURE, TOP_P)
        all_tokens += [token]
        # 使用 tokenizer.decode 将 all_tokens[out_last:] 解码为文本，
        # 并检查解码结果是否包含无效的 utf-8 字符（'\ufffd'）。如果结果有效，则将其打印出来。
        try:
            tmp = tokenizer.decode(all_tokens[out_last:])
            if '\ufffd' not in tmp: # only print when we have a valid utf-8 string
                print(tmp, end="", flush=True)
                out_last = i + 1
        except:
            pass
        # 调用模型的 forward 方法，将生成的 token 和当前的状态传递给模型，获取更新的 out 和 state。
        out, state = model.forward(token, state)       
print('\n')