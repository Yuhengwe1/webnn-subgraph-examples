// buildSubgraph(), subgraph(), and their options are proposed API, not current WebNN.
// Shorthand helpers are described at the end; their bodies are intentionally omitted.

// A struct-like configuration record for this example.
export class GQAConfig {
  constructor({do_rotary = false, scale} = {}) {
    /** @type {boolean} Whether to construct the RoPE region. */
    this.do_rotary = do_rotary;
    /** @type {number|undefined} Original scale; undefined derives 1/sqrt(headSize). */
    this.scale = scale;
  }
}

// Call site:
// const config = new GQAConfig({do_rotary: true, scale: 0.08838});
// const GQA = await defineGQASubgraph(builder, operands, config);
// const {output, present_key, present_value} = builder.subgraph(GQA, operands);

export async function buildGQASubgraph(builder, operands, config) {
  const inputs = {};
  // Create formal inputs from the surrounding operands' descriptors.
  for (const name of Object.keys(operands)) {
    const operand = operands[name];
    inputs[name] = builder.input(name, {
      shape: operand.shape,
      dataType: operand.dataType,
    });
  }
  const outputs = buildGQABody(builder, inputs, config);
  return builder.buildSubgraph(outputs, {name: "GQA", inputs, config});
}

function buildGQABody(builder, inputs, config) {
  // Input descriptors and cache positions.
  let {query, key} = inputs;
  const {value, past_key, past_value, seqlens_k, total_sequence_length} = inputs;
  const [batchSize, numQueryHeads, queryLength, headSize] = query.shape;
  const [, numKVHeads, maxLength] = past_key.shape;
  const cachePositions = buildCachePositions(builder, seqlens_k, queryLength);

  // RoPE (Optional).
  if (config.do_rotary) {
    const rotaryPositions = inputs.position_ids ?? cachePositions;
    query = buildRoPE(builder, query, rotaryPositions, inputs.cos_cache, inputs.sin_cache);
    key = buildRoPE(builder, key, rotaryPositions, inputs.cos_cache, inputs.sin_cache);
  }

  // KV Cache Update
  const cacheIndices = buildCacheIndices(builder, cachePositions, numKVHeads);
  const present_key = builder.scatterND(past_key, cacheIndices, key);
  const present_value = builder.scatterND(past_value, cacheIndices, value);

  // Repeat KV Heads
  const repeatedKey = repeatKVHeads(builder, present_key, numQueryHeads);
  const repeatedValue = repeatKVHeads(builder, present_value, numQueryHeads);

  // Attention Core
  const transposedKey = builder.transpose(repeatedKey, {permutation: [0, 1, 3, 2]});
  // Config can retain an original scale that cannot be recovered after FP16 rounding.
  const scaleValue = config.scale ?? 1 / Math.sqrt(headSize);
  const scale = scalarConstant(builder, query.dataType, scaleValue);
  const scores = builder.mul(builder.matmul(query, transposedKey), scale);
  const mask = buildCausalMask(builder, cachePositions, total_sequence_length, maxLength, query.dataType);
  const probabilities = builder.softmax(builder.add(scores, mask), 3);
  const attention = builder.matmul(probabilities, repeatedValue);

  // Output layout.
  const output = builder.reshape(
    builder.transpose(attention, {permutation: [0, 2, 1, 3]}),
    [batchSize, queryLength, numQueryHeads * headSize]);
  return {output, present_key, present_value};
}

function repeatKVHeads(builder, cache, numQueryHeads) {
  const [batchSize, numKVHeads, maxLength, headSize] = cache.shape;
  const groupSize = numQueryHeads / numKVHeads;
  const grouped = builder.reshape(cache, [batchSize, numKVHeads, 1, maxLength, headSize]);
  return builder.reshape(
    builder.expand(grouped, [batchSize, numKVHeads, groupSize, maxLength, headSize]),
    [batchSize, numQueryHeads, maxLength, headSize]);
}

// Full-head, split-half RoPE only, to illustrate the config-dependent region.
function buildRoPE(builder, operand, positions, cosCache, sinCache) {
  const [batchSize, , sequenceLength, headSize] = operand.shape;
  const cacheShape = [batchSize, 1, sequenceLength, headSize / 2];
  const cosine = builder.reshape(builder.gather(cosCache, positions, {axis: 0}), cacheShape);
  const sine = builder.reshape(builder.gather(sinCache, positions, {axis: 0}), cacheShape);
  const [firstHalf, secondHalf] = builder.split(operand, 2, {axis: 3});
  return builder.concat([
    builder.sub(builder.mul(firstHalf, cosine), builder.mul(secondHalf, sine)),
    builder.add(builder.mul(secondHalf, cosine), builder.mul(firstHalf, sine)),
  ], 3);
}

// Pseudocode shorthand, not new WebNN operations:
// buildCachePositions(builder, seqlens_k, queryLength): signed subtract/add with a token range;
//   cachePositions[batch, token] = seqlens_k[batch] + 1 - queryLength + token.
// buildCacheIndices(builder, cachePositions, numKVHeads): expand/concat indices
//   [batch, KV head, cachePositions[batch, token]], shape [batchSize,numKVHeads,queryLength,3].
// buildCausalMask(builder, cachePositions, totalSequenceLength, maxLength, dataType):
//   visible = keyIndex <= cachePositions AND keyIndex < totalSequenceLength;
//   use where to produce 0 for visible keys, -Infinity otherwise;
//   shape [batchSize,1,queryLength,maxLength].
// scalarConstant(builder, dataType, value): builder.constant with typed scalar data.
