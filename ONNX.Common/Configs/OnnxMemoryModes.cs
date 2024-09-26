using System;

namespace ONNX.Common.Configs
{
    [Flags]
    public enum OnnxMemoryModes
    {
        None,
        DeferLoading,
        UnloadAfterUse,
    }
}
