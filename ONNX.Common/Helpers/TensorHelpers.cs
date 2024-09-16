using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using ONNX.Common.Tensor;

namespace ONNX.Common.Helpers
{
    public static class TensorHelpers
    {
        
        public static ManagedTensor<T> CreateAndFillTensor<T>(T fill, ReadOnlySpan<int> dimensions)
            where T : unmanaged
        {
            var tensor = new ManagedTensor<T>(dimensions, initialize: false);
            
            tensor.SNTensor.Fill(fill);
            
            return tensor;
        }
        
        public static ManagedTensor<T> SoftMax<T>(this ManagedTensor<T> tensor)
            where T: unmanaged, IExponentialFunctions<T>
        {
            return SystemNumericsTensor.SoftMax<T>(tensor.SNTensor);
        }
        
        public static ManagedTensor<T> SoftMaxInPlace<T>(this ManagedTensor<T> tensor)
            where T: unmanaged, IExponentialFunctions<T>
        {
            var snTensor = tensor.SNTensor;
            
            SystemNumericsTensor.SoftMax<T>(snTensor, snTensor);
            
            return tensor;
        }
        
        private readonly struct TopKSession
        {
            public readonly InferenceSession Model;

            public readonly ManagedTensor<long> KInputBuffer;

            public TopKSession()
            {
                Model = new(
                    ResourceHelpers.GetResourceBytes(
                        typeof(TensorHelpers).Assembly, 
                        "topk.onnx")!);

                KInputBuffer = new(
                    (ReadOnlySpan<nint>) [ 1 ], 
                    initialize: false,
                    pinned: true);
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
        
        public readonly struct TopKOutput(ManagedTensor<float> logits, ManagedTensor<long> indices)
        {
            public readonly ManagedTensor<float> Logits = logits;
            
            public readonly ManagedTensor<long> Indices = indices;
        }
        
        public static TopKOutput TopK(this ManagedTensor<float> logitsInput, ulong k, bool pinned = false)
        {
            // https://josephrocca.github.io/onnxscript-editor/demo/
            
            // Code for generating TopK ONNX model: https://github.com/budgetdevv/FlorenceSharp/blob/e2860af0f173775a14e81e3b4b3dfde403d32f20/OnnxExtensions/main.py#L126
            
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#TopK
            
            var dimensions = logitsInput.SNTensor.Lengths.ToArray();
            
            // The last dimension should be K.
            
            dimensions[^1] = unchecked((int) k);
            
            var tensorDimensions = (TensorDimensions) dimensions;
            
            // Create new output buffers

            var logitsOutput = new ManagedTensor<float>(tensorDimensions, initialize: false, pinned);
            
            var indicesOutput = new ManagedTensor<long>(tensorDimensions, initialize: false, pinned);

            var topK = TopKSessionSessionCurrentThread;
            
            var topKModel = topK.Model;
            
            var kInputBuffer = topK.KInputBuffer;

            // Probably the fastest way to store its value
            MemoryMarshal.GetArrayDataReference(kInputBuffer.ValuesArr) = unchecked((long) k);

            topKModel.Run(
                inputs: 
                [
                    logitsInput.AsNamedOnnxValue("logits"),
                    kInputBuffer.AsNamedOnnxValue("k"),
                ], 
                outputs:
                [ 
                    logitsOutput.AsNamedOnnxValue("values"),
                    indicesOutput.AsNamedOnnxValue("indices"),
                ]
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

        public static NamedOnnxValue AsNamedOnnxValue<T>(this DenseTensor<T> tensor, string name)
            where T : unmanaged
        {
            return NamedOnnxValue.CreateFromTensor(name, tensor);
        }
        
        public static NamedOnnxValue AllocateEmptyNamedTensorValue<T>(string name)
            where T : unmanaged
        {
            var tensor = new DenseTensor<T>(Array.Empty<T>().AsMemory(), [ 0 ]);
            
            return NamedOnnxValue.CreateFromTensor(name, tensor);
        }
    }
}