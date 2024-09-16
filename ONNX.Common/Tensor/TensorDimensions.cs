using System;

namespace ONNX.Common.Tensor
{
    public readonly struct TensorDimensions
    {
        public readonly nint[] DimensionsNint;
        
        public readonly int[] DimensionsInt;

        public TensorDimensions(nint[] dimensions)
        {
            DimensionsNint = dimensions;
            
            var length = dimensions.Length;
            
            var dimensionsInt = DimensionsInt = new int[length];
            
            for (int i = 0; i < length; i++)
            {
                dimensionsInt[i] = (int) dimensions[i];
            }
        }

        public TensorDimensions(int[] dimensions)
        {
            DimensionsInt = dimensions;
            
            var length = dimensions.Length;
            
            var dimensionsNint = DimensionsNint = new nint[length];
            
            for (int i = 0; i < length; i++)
            {
                dimensionsNint[i] = dimensions[i];
            }
        }
        
        public static implicit operator nint[](TensorDimensions dimensions)
        {
            return dimensions.DimensionsNint;
        }
        
        public static implicit operator int[](TensorDimensions dimensions)
        {
            return dimensions.DimensionsInt;
        }
        
        public static implicit operator TensorDimensions(nint[] dimensions)
        {
            return new(dimensions);
        }
        
        public static implicit operator TensorDimensions(int[] dimensions)
        {
            return new(dimensions);
        }
        
        public static implicit operator ReadOnlySpan<nint>(TensorDimensions dimensions)
        {
            return dimensions.DimensionsNint;
        }
        
        public static implicit operator ReadOnlySpan<int>(TensorDimensions dimensions)
        {
            return dimensions.DimensionsInt;
        }
        
        public static implicit operator TensorDimensions(ReadOnlySpan<nint> dimensions)
        {
            return new(dimensions.ToArray());
        }
        
        public static implicit operator TensorDimensions(ReadOnlySpan<int> dimensions)
        {
            return new(dimensions.ToArray());
        }
    }
}