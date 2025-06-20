using System.Numerics;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;

namespace ONNX.Common.Helpers
{
    public static class TensorHelpers
    {
        public static Tensors.ManagedTensor<T> CreateAndFillTensor<T>(
            T fill,
            ReadOnlySpan<nint> dimensions)
            where T: unmanaged
        {
            var tensor = new Tensors.ManagedTensor<T>(dimensions, initialize: false);

            tensor.ValuesArr.AsSpan().Fill(fill);

            return tensor;
        }
        
        public static Tensors.ManagedTensor<T> SoftMaxInPlace<T>(this Tensors.ManagedTensor<T> tensor)
            where T: unmanaged, IExponentialFunctions<T>
        {
            var snTensor = tensor.Tensor;
            
            SystemNumericsTensor.SoftMax<T>(snTensor, snTensor);
            
            return tensor;
        }
        
        private readonly struct TopKSession
        {
            public readonly InferenceSession Model;

            public readonly Tensors.ManagedTensor<long> KInputBuffer;

            public TopKSession()
            {
                Model = new(ResourceHelpers.GetResourceBytes(
                    typeof(TensorHelpers).Assembly,
                    "topk.onnx")!
                );

                KInputBuffer = new(
                    (ReadOnlySpan<nint>) [ 1 ], 
                    initialize: false,
                    pinned: true
                );
            }
        }
        
        [ThreadStatic]
        private static TopKSession? TopKSessionSessionThreadStatic;

        private static TopKSession TopKSessionSessionCurrentThread
        {
            get
            {
                return TopKSessionSessionThreadStatic ?? CreateAndSetTopK();

                [MethodImpl(MethodImplOptions.NoInlining)]
                TopKSession CreateAndSetTopK()
                {
                    return (TopKSessionSessionThreadStatic = new TopKSession()).GetValueOrDefault();
                }
            }
        }
        
        public readonly struct TopKOutput(Tensors.ManagedTensor<float> logits, Tensors.ManagedTensor<long> indices)
        {
            public readonly Tensors.ManagedTensor<float> Logits = logits;
            
            public readonly Tensors.ManagedTensor<long> Indices = indices;
        }
        
        public static TopKOutput TopK(this Tensors.ManagedTensor<float> logitsInput, ulong k, bool pinned = false)
        {
            // https://josephrocca.github.io/onnxscript-editor/demo/
            
            // Code for generating TopK ONNX model: https://github.com/budgetdevv/FlorenceSharp/blob/e2860af0f173775a14e81e3b4b3dfde403d32f20/OnnxExtensions/main.py#L126
            
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#TopK

            var dimensions = logitsInput.Dimensions.ToArray();
            
            // The last dimension should be K.
            
            dimensions[^1] = unchecked((int) k);
            
            // Create new output buffers

            var logitsOutput = new Tensors.ManagedTensor<float>(dimensions, initialize: false, pinned);
            
            var indicesOutput = new Tensors.ManagedTensor<long>(dimensions, initialize: false, pinned);

            var topK = TopKSessionSessionCurrentThread;
            
            var topKModel = topK.Model;
            
            var kInputBuffer = topK.KInputBuffer;

            // Probably the fastest way to store its value
            MemoryMarshal.GetArrayDataReference(kInputBuffer.ValuesArr) = unchecked((long) k);

            using var ioBindings = topKModel.CreateIoBinding();

            logitsInput.BindAsInput(ioBindings, "logits");

            kInputBuffer.BindAsInput(ioBindings, "k");

            logitsOutput.BindAsOutput(ioBindings, "values");

            indicesOutput.BindAsOutput(ioBindings, "indices");

            topKModel.RunWithBinding(
                runOptions: ONNXConstants.DEFAULT_RUN_OPTIONS,
                ioBinding: ioBindings
            );
            
            return new(logitsOutput, indicesOutput);
        }

        public static T GetDimensionSize<T>(this ReadOnlySpan<T> dimensions)
            where T: unmanaged, IMultiplyOperators<T, T, T>
        {
            var length = dimensions.Length;
            
            if (length != 0)
            {
                ref var currentValue = ref MemoryMarshal.GetReference(dimensions);
            
                ref var lastValueOffsetByOne = ref Unsafe.Add(ref currentValue, length);
            
                var accumulator = currentValue;

                if (length != 1)
                {
                    for (currentValue = ref Unsafe.Add(ref currentValue, 1); 
                         !Unsafe.AreSame(ref currentValue, ref lastValueOffsetByOne); 
                         currentValue = ref Unsafe.Add(ref currentValue, 1))
                    {
                        accumulator *= currentValue;
                    }
                }
            
                return accumulator;
            }

            return default;
        }
    }
}