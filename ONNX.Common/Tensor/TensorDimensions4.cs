using System;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;

namespace ONNX.Common.Tensor
{
    [InlineArray(4)]
    public struct TensorDimensionInt4
    {
        private int _0;
            
        public TensorDimensionInt4(int _0, int _1, int _2, int _3)
        {
            this._0 = _0;
            this[1] = _1;
            this[2] = _2;
            this[3] = _3;
        }
        
        [UnscopedRef]
        public ReadOnlySpan<int> GetDimensionSpan()
        {
            return this;
        }
    }

    [InlineArray(4)]
    public struct TensorDimensionNInt4
    {
        private nint _0;
            
        public TensorDimensionNInt4(nint _0, nint _1, nint _2, nint _3)
        {
            this._0 = _0;
            this[1] = _1;
            this[2] = _2;
            this[3] = _3;
        }
        
        [UnscopedRef]
        public ReadOnlySpan<nint> GetDimensionSpan()
        {
            return this;
        }
    }
    
}