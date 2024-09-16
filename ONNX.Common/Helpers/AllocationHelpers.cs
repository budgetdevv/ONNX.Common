using System;

namespace ONNX.Common.Helpers
{
    public static class AllocationHelpers
    {
        public static T[] AllocatePinnedArrayUninitialized<T>(int length)
        {
            return AllocatePinnedArrayUninitialized<T>(unchecked((uint) length));
        }
        
        public static T[] AllocatePinnedArrayUninitialized<T>(uint length)
        {
            return GC.AllocateUninitializedArray<T>(
                unchecked((int) length), 
                pinned: true);
        }
        
        public static T[] AllocatePinnedArrayAndFill<T>(int length, T fillValue)
        {
            return AllocatePinnedArrayAndFill<T>(unchecked((uint) length), fillValue);
        }
        
        public static T[] AllocatePinnedArrayAndFill<T>(uint length, T fillValue)
        {
            var arr = AllocatePinnedArrayUninitialized<T>(length);
            
            arr.AsSpan().Fill(fillValue);

            return arr;
        }
    }
}