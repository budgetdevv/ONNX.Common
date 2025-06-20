using System.Text;

namespace ONNX.Common.Helpers
{
    public static class DebugHelpers
    {
        public static string GetSpanPrintString<T>(this ReadOnlySpan<T> span)
        {
            return span.ToArray().GetArrPrintString();
        }
        
        public static string GetArrPrintString<T>(this T[] arr)
        {
            // It has to include commas and the array brackets as well...
            var stringBuilder = new StringBuilder(arr.Length * 2);

            stringBuilder.Append('[');
            
            const string SEPARATOR = ", ";
            
            foreach (var item in arr)
            {
                stringBuilder.Append(item);
                stringBuilder.Append(SEPARATOR);
            }
            
            var separatorLength = SEPARATOR.Length;
            stringBuilder.Remove(stringBuilder.Length - separatorLength, separatorLength);
            
            stringBuilder.Append(']');
            
            return stringBuilder.ToString();
        }
    }
}