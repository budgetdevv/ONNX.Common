﻿using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using System.Text;
using ONNX.Common;
using ONNX.Common.Configs;
using ONNX.Common.Helpers;
using ONNX.Common.Tensor;
using Tokenizers.NET;

namespace Playground
{
    internal static class Program
    {
        private struct OnnxConfig: ConfigurableOnnxModel.IConfig
        {
            private static readonly ConfigurableOnnxModel.BuiltConfig CONFIG = 
                new ConfigurableOnnxModel.ConfigBuilder()
                    .WithBackendType(BackendType.CPU)
                    .WithMemoryMode(OnnxMemoryModes.None)
                    .WithRegisterOrtExtensions()
                    .WithModelPath("/Users/trumpmcdonaldz/Desktop/JINA/model_quantized.onnx")
                    .Build();

            public static ConfigurableOnnxModel.BuiltConfig Config => CONFIG;
        }

        private struct JinaReranker
        {
            private struct TokenizerConfig: ITokenizerConfig
            {
                public static uint ExpectedMaxInputLength => 512;

                public static uint ExpectedMaxBatches => 2;

                public static string TokenizerJsonPath => "Resources/jina_tokenizer.json";
            }

            public readonly struct Output(int index, float score)
            {
                public readonly int Index = index;

                public readonly float Score = score;
            }
            
            private Tokenizer<TokenizerConfig> Tokenizer;
            
            private ConfigurableOnnxModel<OnnxConfig> Model;

            public JinaReranker()
            {
                Tokenizer = new();
                Model = new();
            }
            
            // Very messy and suboptimal but it works
            public Output[] Rerank(string query, params ReadOnlySpan<string> inputs)
            {
                // Slow but whatever
                inputs = inputs
                    .ToArray()
                    .Select(input => $"<s> {query}</s></s> {input}</s>")
                    .ToArray();
                
                ref var tokenizer = ref Tokenizer;
                
                using var tokenizeOutputs = tokenizer.TokenizeBatch(inputs, addSpecialTokens: false);

                var tokenizeOutputSpan = tokenizeOutputs.Buffer.AsSpan();

                var firstOutput = tokenizeOutputSpan[0];
                
                // Console.WriteLine(Encoding.UTF8.GetString(tokenizer.Decode(firstOutput.IDs, false).TextBuffer.AsReadOnlySpan()));
                //
                // Console.WriteLine(firstOutput.IDs.AsReadOnlySpan().GetSpanPrintString());
                
                var numInputs = inputs.Length;
                
                var dims = (TensorDimensions) (ReadOnlySpan<int>) [ numInputs, (int) firstOutput.IDs.Length ];
                
                var idTensor = new ManagedTensor<long>(
                    dims, 
                    initialize: false,
                    pinned: true);

                var snIDTensor = idTensor.SNTensor;
                
                var attentionMaskTensor = new ManagedTensor<long>(
                    dims, 
                    initialize: false,
                    pinned: true);

                var snAttentionMaskTensor = attentionMaskTensor.SNTensor;

                var currentBatchIndex = 0;
                
                foreach (var output in tokenizeOutputSpan)
                {
                    var currentBatchIndexPlusOne = currentBatchIndex + 1;

                    ReadOnlySpan<NRange> range = [currentBatchIndex..currentBatchIndexPlusOne, NRange.All];
                    
                    // End index is exclusive
                    // https://sharplab.io/#v2:EYLgxg9gTgpgtADwGwBYA0AXEUCuA7AHwAEAmARgFgAoagNwEMoACAZwAd68mBeJgCgBKMegBMA8ngA2ATwDKHPAB4AlngwA+AJRMA2kgCcaJiQC6AbmrUA9FaYBRPCKaqRMBM5ZM3YSThbLaGDpGVkllMBgyHlYFHQAGADoEsnNLKiIyfT4WMIiyeJNNC3TM7NzIhIAZGDwAcwwACyKgA==
                    var idSlice = snIDTensor[range];
                    var attentionMaskSlice = snAttentionMaskTensor[range];
                    
                    // Slow but whatever
                    using var ids = output.IDs.Widen();
                    
                    using var attentionMask = output.AttentionMask.Widen();

                    var idSpan = ids.Buffer.Cast<long>().AsSpan();
                    var attentionMaskSpan = attentionMask.Buffer.Cast<long>().AsSpan();
                    
                    idSpan.CopyTo(MemoryMarshal.CreateSpan(
                        ref idSlice.GetPinnableReference(),
                        (int) idSlice.FlattenedLength)
                    );
                    
                    attentionMaskSpan.CopyTo(MemoryMarshal.CreateSpan(
                        ref attentionMaskSlice.GetPinnableReference(),
                        (int) attentionMaskSlice.FlattenedLength)
                    );
                    
                    // Unfortunately slicing copies atm
                    snIDTensor[range] = idSlice;
                    snAttentionMaskTensor[range] = attentionMaskSlice;
                        
                    currentBatchIndex = currentBatchIndexPlusOne;
                }
                
                var logitsTensor = new ManagedTensor<float>(
                    (ReadOnlySpan<int>) [ numInputs, 1 ], 
                    initialize: false,
                    pinned: true);
                
                using (var handle = Model.GetSessionHandle())
                {
                    handle.Session.Run(
                        inputs: 
                        [ 
                            idTensor.AsNamedOnnxValue("input_ids"),
                            attentionMaskTensor.AsNamedOnnxValue("attention_mask"),
                        ],
                        outputs: [ logitsTensor.AsNamedOnnxValue("logits") ]
                    );
                    
                    // logitsTensor.PrintTensor();

                    logitsTensor = logitsTensor.Reshape([ numInputs ]);
                    
                    // logitsTensor.PrintTensor();
                    
                    var topK = logitsTensor.TopK((ulong) numInputs);

                    var outputs = new List<Output>(numInputs);
                    
                    var currentIndex = 0;
                    
                    foreach (var index in topK.Indices.ValuesArr)
                    {
                        outputs.Add(new((int) index, topK.Logits.ValuesArr[currentIndex++]));
                    }
                    
                    return outputs.ToArray();
                }
            }
        }
        
        private static void Main(string[] args)
        {
            var ranker = new JinaReranker();

            ReadOnlySpan<string> inputs =
            [
                "Organic skincare for sensitive skin with aloe vera and chamomile.",
                "New makeup trends focus on bold colors and innovative techniques",
                "Bio-Hautpflege für empfindliche Haut mit Aloe Vera und Kamille",
                "Neue Make-up-Trends setzen auf kräftige Farben und innovative Techniken",
                "Cuidado de la piel orgánico para piel sensible con aloe vera y manzanilla",
                "Las nuevas tendencias de maquillaje se centran en colores vivos y técnicas innovadoras",
                "针对敏感肌专门设计的天然有机护肤产品",
                "新的化妆趋势注重鲜艳的颜色和创新的技巧",
                "敏感肌のために特別に設計された天然有機スキンケア製品",
                "新しいメイクのトレンドは鮮やかな色と革新的な技術に焦点を当てています",
            ];
            
            var outputs = ranker.Rerank(
                query: "Organic skincare products for sensitive skin", 
                inputs
            );
            
            foreach (var output in outputs)
            {
                var logit = output.Score;

                var index = output.Index;
                
                Console.WriteLine(
                $"""
                Text: {inputs[index]} [ {index} ]
                Logit Value: {logit}
                Score ( Sigmoid ): {Sigmoid(logit)}
                
                """);
            }

            return;
            
            static float Sigmoid(float x)
            {
                return 1.0f / (1.0f + MathF.Exp(-x));
            }
        }
    }
}