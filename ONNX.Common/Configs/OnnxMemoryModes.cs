using System;

namespace ONNX.Common.Configs
{
    [Flags]
    public enum OnnxMemoryModes
    {
        None,
        DeferLoading,
        // UnloadAfterUse is also implicitly DeferLoading.
        // It does make sense because model reloading is deferred.
        UnloadAfterUse,
    }
}
