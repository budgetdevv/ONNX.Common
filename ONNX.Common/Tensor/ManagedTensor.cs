using System;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using ONNX.Common.Helpers;

namespace ONNX.Common.Tensor
{
    public readonly struct ManagedTensor<T> where T: unmanaged
    {
        public readonly T[] ValuesArr;
        
        public readonly SystemNumericsTensors.Tensor<T> SNTensor;

        public readonly DenseTensor<T> OnnxDenseTensor;

        private static readonly FieldInfo SYSTEM_NUMERICS_TENSOR_VALUES_FIELD_INFO = typeof(SystemNumericsTensors.Tensor<T>)
            .GetField("_values", BindingFlags.NonPublic | BindingFlags.Instance)!;
        
        public ManagedTensor(TensorDimensions dimensions, bool initialize, bool pinned = false)
            :this(initialize ? 
                SystemNumericsTensor.Create<T>(dimensions, pinned) : 
                SystemNumericsTensor.CreateUninitialized<T>(dimensions, pinned),
                dimensions) { }

        public ManagedTensor(SystemNumericsTensors.Tensor<T> snTensor): this(snTensor, snTensor.Lengths) { }
        
        public ManagedTensor(SystemNumericsTensors.Tensor<T> snTensor, TensorDimensions dimensions)
        {
            SNTensor = snTensor;

            #if NET9_0_OR_GREATER
            var arr = GetValuesArray(snTensor);
            #else
            var arr = Unsafe.As<T[]>(SYSTEM_NUMERICS_TENSOR_VALUES_FIELD_INFO.GetValue(snTensor))!;
            #endif

            ValuesArr = arr;
            
            // OnnxORTValue = OrtValue.CreateTensorValueFromMemory<T>(pinnedMemory, dimensions.WidenDimensions());

            Memory<T> memory;
            
            memory = snTensor.IsPinned ? 
                MemoryMarshal.CreateFromPinnedArray(arr, 0, arr.Length) :
                arr.AsMemory();
            
            // Span overload doesn't wrap memory ( Unsurprisingly )
            OnnxDenseTensor = new(memory, dimensions);
            
            return;
            
            [UnsafeAccessor(UnsafeAccessorKind.Field, Name = "_values")]
            static extern ref T[] GetValuesArray(SystemNumericsTensors.Tensor<T> tensor);
        }

        public DenseTensor<T> GetReinterpretedDenseTensorUnsafely(scoped ReadOnlySpan<int> dimensions)
        {
            // Unfortunately DenseTensor<T>'s ctor does check for length.
            var memory = ValuesArr.AsMemory(0, dimensions.GetDimensionSize());
            
            return new(memory, dimensions);
        }
        
        public static ManagedTensor<T> CopyFromDenseTensor(DenseTensor<T> tensor, bool pinned = false)
        {
            // Unfortunately it is impossible to wrap DenseTensor, since ManagedTensor support
            // System.Numerics.Tensors.Tensor<T> as well, which is backed by an actual array.
            
            // Potential solution for avoiding copies: 
            // Pre-allocate pinned ManagedTensor<T> and use it as output for InferenceSession.Run()
            // Downside is having to manually compute output dimensions.
            
            return SystemNumericsTensor.Create<T>(
                tensor.Buffer.ToArray(), 
                (TensorDimensions) tensor.Dimensions, 
                pinned);
        }
        
        public NamedOnnxValue AsNamedOnnxValue(string name)
        {
            return NamedOnnxValue.CreateFromTensor(name, OnnxDenseTensor);
        }
        
        public static implicit operator SystemNumericsTensors.Tensor<T>(ManagedTensor<T> tensor)
        {
            return tensor.SNTensor;
        }
        
        public static implicit operator DenseTensor<T>(ManagedTensor<T> tensor)
        {
            return tensor.OnnxDenseTensor;
        }
        
        public static implicit operator ManagedTensor<T>(SystemNumericsTensors.Tensor<T> tensor)
        {
            return new(tensor);
        }
    }
}