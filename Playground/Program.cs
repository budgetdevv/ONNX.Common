using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using NoParamlessCtor.Shared.Attributes;
using ONNX.Common;
using ONNX.Common.Configs;
using ONNX.Common.Helpers;
using ONNX.Common.Tensor;
using ONNX.Common.Tensors;
using Tokenizers.NET;
using Tokenizers.NET.Helpers;

namespace Playground
{
    internal static partial class Program
    {
        private const string MODEL_PATH = "/Users/trumpmcdonaldz/Desktop/JINA/model_quantized.onnx";

        [ModuleInitializer]
        internal static void Init()
        {
            RuntimeHelpers.RunClassConstructor(typeof(JinaReranker).TypeHandle);
        }
        
        private static async Task Main(string[] args)
        {
            // await CheckCodegen();

            await SampleInference();
        }
        
        private static async Task CheckCodegen()
        {
            // How to check codegen:
            // Mac:
            // export DOTNET_JitDisasm="*_DISASM"
            // Windows:
            // $Env:DOTNET_JitDisasm="*_DISASM"
            // dotnet run -c Release
            
            var model = await JinaReranker.InitializeAsync(MODEL_PATH);

            // Ensure we ain't cheating by passing a constant span value
            // E.x. TokenizeBatch_DISASM(model.Tokenizer, [ "Hi", "Bye" ]);
            var list = new List<string>()
            {
                "Organic skincare for sensitive skin with aloe vera and chamomile.",
                "New makeup trends focus on bold colors and innovative techniques",
            };
            
            DisposeSessionHandle_DISASM(GetSessionHandle_DISASM(model.Model));
        }
        
        private const MethodImplOptions DISASM_METHOD_IMPL_OPTIONS = MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization;
        
        [MethodImpl(DISASM_METHOD_IMPL_OPTIONS)]
        private static ConfigurableOnnxModel<JinaRerankerONNXConfig>.SessionHandle GetSessionHandle_DISASM(ConfigurableOnnxModel<JinaRerankerONNXConfig> model)
        {
            return model.GetSessionHandle();
        }
        
        [MethodImpl(DISASM_METHOD_IMPL_OPTIONS)]
        private static void DisposeSessionHandle_DISASM(ConfigurableOnnxModel<JinaRerankerONNXConfig>.SessionHandle handle)
        {
            handle.Dispose();
        }
        
        private readonly struct JinaRerankerONNXConfig: ConfigurableOnnxModel.IConfig
        {
            public static ConfigurableOnnxModel.ConfigBuilder ConfigBuilder =>
                new ConfigurableOnnxModel.ConfigBuilder()
                    .WithBackendType(BackendType.CPU)
                    .WithMemoryMode(OnnxMemoryModes.DeferLoading)
                    .WithRegisterOrtExtensions();
        }

        [NoParamlessCtor]
        private partial struct JinaReranker(Tokenizer tokenizer, ConfigurableOnnxModel<JinaRerankerONNXConfig> model)
        {
            public readonly struct Output(int index, float score)
            {
                public readonly int Index = index;

                public readonly float Score = score;
            }

            internal Tokenizer Tokenizer = tokenizer;
            
            internal ConfigurableOnnxModel<JinaRerankerONNXConfig> Model = model;

            public static async ValueTask<JinaReranker> InitializeAsync(string onnxModelPath)
            {
                // var tokenizer = (await new TokenizerBuilder()
                //     .SetExpectedMaxInputLength(512)
                //     .SetExpectedMaxBatches(16)
                //     .SetExceedExpectedMaxBatchesBehavior(ExceedExpectedMaxBatchesBehavior.AllocateBuffer)
                //     .DownloadFromHuggingFaceRepoAsync("jinaai/jina-reranker-v2-base-multilingual"))
                //     .Build();

                var tokenizer = new TokenizerBuilder()
                    .SetExpectedMaxInputLength(512)
                    .SetExpectedMaxBatches(16)
                    .SetExceedExpectedMaxBatchesBehavior(ExceedExpectedMaxBatchesBehavior.AllocateBuffer)
                    .SetTokenizerJsonPath("Resources/jina_tokenizer.json")
                    .Build();

                var model = new ConfigurableOnnxModel<JinaRerankerONNXConfig>(onnxModelPath);

                return new JinaReranker(tokenizer, model);
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

                // foreach (var output in tokenizeOutputSpan)
                // {
                //     Console.WriteLine(output.IDs.Length);
                // }

                var firstOutput = tokenizeOutputSpan[0];
                
                // Console.WriteLine(Encoding.UTF8.GetString(tokenizer.Decode(firstOutput.IDs, false).TextBuffer.AsReadOnlySpan()));
                //
                // Console.WriteLine(firstOutput.IDs.AsReadOnlySpan().GetSpanPrintString());
                
                var numInputs = inputs.Length;

                var dims = (ReadOnlySpan<nint>) [ numInputs, (nint) firstOutput.IDs.Length ];
                
                var idTensor = new ManagedTensor<long>(
                    dims, 
                    initialize: false,
                    pinned: true
                );

                var snIDTensor = idTensor.Tensor;
                
                var attentionMaskTensor = new ManagedTensor<long>(
                    dims, 
                    initialize: false,
                    pinned: true
                );

                var snAttentionMaskTensor = attentionMaskTensor.Tensor;

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
                        (int) idSlice.FlattenedLength
                    ));
                    
                    attentionMaskSpan.CopyTo(MemoryMarshal.CreateSpan(
                        ref attentionMaskSlice.GetPinnableReference(),
                        (int) attentionMaskSlice.FlattenedLength
                    ));
                    
                    // Unfortunately slicing copies atm
                    snIDTensor[range] = idSlice;
                    snAttentionMaskTensor[range] = attentionMaskSlice;
                        
                    currentBatchIndex = currentBatchIndexPlusOne;
                }
                
                var logitsTensor = new ManagedTensor<float>(
                    [ numInputs, 1 ],
                    initialize: false,
                    pinned: true);

                using (var handle = Model.GetSessionHandle())
                {
                    var session = handle.Session;

                    using var binding = session.CreateIoBinding();

                    idTensor.BindAsInput(binding, "input_ids");

                    attentionMaskTensor.BindAsInput(binding, "attention_mask");

                    logitsTensor.BindAsOutput(binding, "logits");

                    session.RunWithBinding(
                        runOptions: new RunOptions(),
                        ioBinding: binding
                    );

                    logitsTensor.Print();

                    // logitsTensor.Reshape([ numInputs ]);

                    logitsTensor.Squeeze();

                    logitsTensor.Print();

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
        
        private static async Task SampleInference()
        {
            var reRanker = await JinaReranker.InitializeAsync(MODEL_PATH);

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
            
            var outputs = reRanker.Rerank(
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