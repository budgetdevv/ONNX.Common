using System.Diagnostics;
using System.Numerics.Tensors;
using System.Reflection;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using NoParamlessCtor.Shared.Attributes;
using ONNX.Common.Helpers;

namespace ONNX.Common.Tensors
{
    [NoParamlessCtor]
    public partial struct ManagedTensor<T>: IDisposable
        where T: unmanaged
    {
        private const string TENSOR_VALUES_ARRAY_FIELD_NAME = "_values";

        public Tensor<T> Tensor;

        public OrtValue OrtValue;

        public T[] ValuesArr => GetValuesArrayUnsafely(Tensor);

        public ReadOnlySpan<nint> Dimensions => Tensor.Lengths;

        #if !NET9_0_OR_GREATER
        private static readonly FieldInfo SYSTEM_NUMERICS_TENSOR_VALUES_FIELD_INFO = typeof(Tensor<T>)
            .GetField(TENSOR_VALUES_ARRAY_FIELD_NAME, BindingFlags.NonPublic | BindingFlags.Instance)!;
        #endif

        public ManagedTensor(ReadOnlySpan<nint> dimensions, bool initialize, bool pinned = false): this(
            initialize ?
            SystemNumericsTensor.Create<T>(dimensions, pinned) :
            SystemNumericsTensor.CreateUninitialized<T>(dimensions, pinned)
        ) { }

        public ManagedTensor(Tensor<T> tensor): this(tensor, tensor.Lengths) { }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private ManagedTensor(Tensor<T> tensor, ReadOnlySpan<nint> dimensions)
        {
            Debug.Assert(tensor.IsDense);

            Tensor = tensor;

            OrtValue = CreateOrtValue(tensor, dimensions);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe OrtValue CreateOrtValue(Tensor<T> tensor, ReadOnlySpan<nint> dimensions)
        {
            var arr = GetValuesArrayUnsafely(tensor);

            var memory = tensor.IsPinned ?
                MemoryMarshal.CreateFromPinnedArray(arr, 0, arr.Length) :
                arr.AsMemory();

            var dimensionsLength = dimensions.Length;

            long[] ortDims;

            // This branch is elided
            if (sizeof(nint) == sizeof(long))
            {
                ortDims = new long[dimensionsLength];

                ref var arrStart = ref Unsafe.As<long, nint>(
                    ref MemoryMarshal.GetArrayDataReference(ortDims)
                );

                dimensions.CopyTo(MemoryMarshal.CreateSpan(
                    ref arrStart, dimensionsLength
                ));
            }

            else
            {
                // x86 is shit slow anyway

                ortDims = Array.ConvertAll(
                    dimensions.ToArray(),
                    x => (long) x
                );
            }

            // Yes, there's OrtValue.CreateTensorValueFromSystemNumericsTensorObject(),
            // but we use special hack above to quickly construct a pinned Memory<T>,
            // should the Tensor itself be pinned ( The Tensor pulls from POH when pinned )
            return OrtValue.CreateTensorValueFromMemory(
                OrtMemoryInfo.DefaultInstance,
                memory,
                ortDims
            );
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static T[] GetValuesArrayUnsafely(Tensor<T> tensor)
        {
            #if NET9_0_OR_GREATER
            return GetValuesArray(tensor);

            [UnsafeAccessor(UnsafeAccessorKind.Field, Name = TENSOR_VALUES_ARRAY_FIELD_NAME)]
            static extern ref T[] GetValuesArray(Tensor<T> tensor);
            #else
            return Unsafe.As<T[]>(SYSTEM_NUMERICS_TENSOR_VALUES_FIELD_INFO.GetValue(tensor))!;
            #endif
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void BindAsInput(OrtIoBinding binding, string name)
        {
            binding.BindInput(name, OrtValue);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void BindAsOutput(OrtIoBinding binding, string name)
        {
            binding.BindOutput(name, OrtValue);
        }

        public void Reshape(ReadOnlySpan<nint> newDimensions)
        {
            var tensor = Tensor = Tensor.Reshape(newDimensions);

            OrtValue = CreateOrtValue(tensor, newDimensions);
        }

        public void Squeeze()
        {
            var tensor = Tensor = Tensor.Squeeze();

            OrtValue = CreateOrtValue(tensor, tensor.Lengths);
        }

        public void Print()
        {
            Console.WriteLine(ValuesArr.GetArrPrintString());
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static implicit operator Tensor<T>(ManagedTensor<T> tensor)
        {
            return tensor.Tensor;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static implicit operator OrtValue(ManagedTensor<T> tensor)
        {
            return tensor.OrtValue;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static implicit operator T[](ManagedTensor<T> tensor)
        {
            return tensor.ValuesArr;
        }

        public void Dispose()
        {
            OrtValue.Dispose();
        }
    }
}