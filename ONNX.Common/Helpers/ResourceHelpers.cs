using System.IO;
using System.Reflection;

namespace ONNX.Common.Helpers
{
    public static class ResourceHelpers
    {
        public static Stream? GetResourceStream(Assembly assembly, string resourcePath)
        {
            resourcePath = $"{assembly.GetName().Name}.Resources.Embedded.{resourcePath}";
            
            return assembly.GetManifestResourceStream(resourcePath);
        }

        public static byte[]? GetResourceBytes(Assembly assembly, string resourcePath)
        {
            using var stream = GetResourceStream(assembly, resourcePath);
            
            if (stream != null)
            {
                var length = (int) stream.Length;
                
                var buffer = new byte[length];
                
                _ = stream.Read(buffer, 0, length);
                
                return buffer;
            }
            
            return null;
        }
    }
}