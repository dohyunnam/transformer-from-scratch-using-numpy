from _modules import *

def dropout(x, drop_prob):
    mask = binomial(1, 1 - drop_prob, size=x.shape)
    return mask * x / (1 - drop_prob)

def relu(x):
    return maximum(0, x)

def embedding(x, vocab_size, d_model):
    """
    https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
    """
    embedding_matrix = randn(vocab_size, d_model) * sqrt(1/d_model)
    return embedding_matrix[x]

def softmax(x, axis=-1):
    """
    makes all elements add up to zero for stable training
    (making it into a percentage)
    does it for each row
    """
    exp_x = exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def scaled_dot_product(q, k, v, mask=None):
    """ 
    :dot-product (multiplicative) attention:

    MatMul between Q and K
    Scale between Q and K
    Mask (opt.) bewteen Q and K
    Softmax between Q and K
    MatMul output with V 
    """
    scaling_factor = 1 / sqrt(k.shape[-1])
    scaled = matmul(q, k.transpose(0,1,3,2)) * scaling_factor
    if mask is not None: scaled += mask # to block data leakage
    attention = softmax(scaled)
    values = matmul(attention, v)
    return values, attention

def linear(x, in_features, out_features, bias=True):
    """
    referenced https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    output = x * A^T + b
    """
    k = 1 / in_features
    weights = uniform(-sqrt(k), sqrt(k), size=(out_features, in_features))
    output = matmul(x, weights.T)
    if bias is not None: #makes bias optional
        output += uniform(-sqrt(k), sqrt(k), size=(out_features))
    return output

def positional_encoding(d_model, max_sequence_length):
    """
    positional encoding

    uses sin and cos to get even and odd, stacks them together at the end

    PE(pos, 2i) = sin(pos/1000^(10000^(2i/d_model)))
    PE(pos, 2i+1) = cos(pos/1000^(10000^(2i/d_model)))
    """
    even_i = arange(0, d_model, 2)
    denominator = power(10000, even_i/d_model)
    position = arange(max_sequence_length).reshape(max_sequence_length, 1)
    even_PE = sin(position / denominator)
    odd_PE = cos(position / denominator)
    stacked = stack([even_PE, odd_PE], axis=2)
    PE = stacked.reshape(max_sequence_length, -1)
    return PE

def multi_head_attention(x, mask, d_model, num_heads):
    """
    doing multihead attention
    splits matrix into num_head parts
    does attention on each (basically scale_dot_product)
    adds the split up parts together

    MultiHead(Q,K,V) = Concat(Head1...headh)Wo
    where headi = Attention(QWiq , KWik, VWiv)
    """
    head_dim = d_model // num_heads
    batch_size, sequence_length, d_model = x.shape
    qkv = linear(x, d_model, 3 * d_model)
    qkv = qkv.reshape(batch_size, sequence_length, num_heads, 3 * head_dim)
    qkv = qkv.transpose(0, 2, 1, 3)
    q, k, v = split(qkv, 3, axis=-1)
    values, attention = scaled_dot_product(q, k, v, mask)
    values = values.transpose(0, 2, 1, 3).reshape(batch_size, sequence_length, num_heads * head_dim)
    out = linear(values, d_model, d_model)
    return out

def layer_norm(inputs, parameters_shape, eps=1e-5):
    """
    normalizes between each layer (center 0 std_dev 1)
    for stable traininig

    setting eps to an arbitrary low value
    gama & beta are learnable parameters
    """
    gamma = np.ones(parameters_shape)
    beta = np.zeros(parameters_shape)
    dims = tuple(range(inputs.ndim - len(parameters_shape), inputs.ndim))
    mean = inputs.mean(axis=dims, keepdims=True)
    var = ((inputs - mean) ** 2).mean(axis=dims, keepdims=True)
    std = np.sqrt(var + eps)
    y = (inputs - mean) / std
    out = gamma * y + beta
    return out

def position_wise_feed_forward(x, d_model, hidden, drop_prob=0.1):
    """
    linear -> relu -> dropout -> linear

    FFN(x) = max(0, xW1 + b1)W2 + b2 
    """
    x = linear(x, d_model, hidden)
    x = relu(x)
    x = dropout(x, drop_prob)
    x = linear(x, hidden, d_model)
    return x

def sentence_embedding(  max_sequence_length, 
                        d_model, language_to_index,
                        START_TOKEN, END_TOKEN, PADDING_TOKEN,
                        x,
                        start_token,
                        end_token
                    ):
    """
    convert sentences into vectors
    """
    vocab_size = len(language_to_index)
    
    def batch_tokenize(batch, start_token, end_token):

        def tokenize(sentence, start_token, end_token):
            sentence_word_indicies = [language_to_index[token] for token in list(sentence)]
            if start_token:
                sentence_word_indicies.insert(0, language_to_index[START_TOKEN])
            if end_token:
                sentence_word_indicies.append(language_to_index[END_TOKEN])
            for _ in range(len(sentence_word_indicies), max_sequence_length):
                sentence_word_indicies.append(language_to_index[PADDING_TOKEN])
            return np.array(sentence_word_indicies)

        tokenized = []
        for sentence_num in range(len(batch)):
            tokenized.append( tokenize(batch[sentence_num], start_token, end_token) )
        tokenized = stack(tokenized)
        return tokenized
    
    x = batch_tokenize(x, start_token, end_token)
    x = embedding(x, vocab_size, d_model)
    pos = positional_encoding(d_model, max_sequence_length) #assuming im only using cpu model since numpy
    x = dropout(x + pos, 0.1)
    return x

def multi_head_cross_attention(d_model, num_heads, x, y, mask):
    """
    what connects the encoder and decoder
    the two arrows in the transformer diagran
    in between the two models
    """
    head_dim = d_model // num_heads
    batch_size, sequence_length, d_model = x.shape
    kv = linear(x, d_model, 2*d_model)
    q = linear(y, d_model, d_model)
    kv = reshape(kv, (batch_size, sequence_length, num_heads, 2 * head_dim))
    q = reshape(q, (batch_size, sequence_length, num_heads, head_dim))
    kv = transpose(kv, (0, 2, 1, 3))
    q = transpose(q, (0, 2, 1, 3))
    k, v = split(kv, 2, axis=-1)
    values, attention = scaled_dot_product(q, k, v, mask)
    values = reshape(transpose(values, (0, 2, 1, 3)), (batch_size, sequence_length, -1))
    out = linear(values, d_model, d_model)
    return out