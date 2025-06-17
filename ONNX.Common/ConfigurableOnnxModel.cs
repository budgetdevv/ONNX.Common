using System.Diagnostics.CodeAnalysis;
using Microsoft.ML.OnnxRuntime;
using NoParamlessCtor.Shared.Attributes;
using ONNX.Common.Configs;

namespace ONNX.Common
{
    public static partial class ConfigurableOnnxModel
    {
        [NoParamlessCtor]
        public partial struct BuiltConfig
        {
            public SessionOptions SessionOptions;
            
            public BackendType BackendType;

            public int DeviceID;
            
            public OnnxMemoryModes MemoryMode;
            
            public bool RegisterOrtExtensions;
            
            public OrtLoggingLevel LoggingLevel;
            
            internal BuiltConfig(ConfigBuilder configBuilder)
            {
                var sessionOptions = SessionOptions = new();
                
                var backendType = BackendType = configBuilder.BackendType;
                
                var deviceID = DeviceID = configBuilder.DeviceID;
                
                MemoryMode = configBuilder.MemoryMode;
                
                var registerOrtExtensions = RegisterOrtExtensions = configBuilder.RegisterOrtExtensions;
                
                var loggingLevel = LoggingLevel = configBuilder.LoggingLevel;
            
                if (registerOrtExtensions)
                {
                    sessionOptions.RegisterOrtExtensions();
                }

                switch (backendType)
                {
                    // "Unhandled exception. Microsoft.ML.OnnxRuntime.OnnxRuntimeException: [ErrorCode:Fail] Provider CPUExecutionProvider has already been registered."
                    // case DeviceType.CPU:
                    //     sessionOptions.AppendExecutionProvider_CPU();
                    //     break;
                
                    case BackendType.TensorRT:
                        sessionOptions.AppendExecutionProvider_Tensorrt(deviceID);
                        break;
                
                    case BackendType.CUDA:
                        sessionOptions.AppendExecutionProvider_CUDA(deviceID);
                        break;
                
                    case BackendType.DirectML:
                        sessionOptions.AppendExecutionProvider_DML(deviceID);
                        break;
                
                    case BackendType.CoreML:
                        // https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/providers/coreml/coreml_provider_factory.h
                        sessionOptions.AppendExecutionProvider_CoreML();
                        break;
                }

                sessionOptions.LogSeverityLevel = loggingLevel;
            }
        }
        
        public struct ConfigBuilder
        {
            public BackendType BackendType;

            public int DeviceID;
            
            public OnnxMemoryModes MemoryMode;
            
            public bool RegisterOrtExtensions;
            
            public OrtLoggingLevel LoggingLevel;

            public ConfigBuilder()
            {
                BackendType = BackendType.CPU;
                DeviceID = 0;
                MemoryMode = OnnxMemoryModes.None;
                RegisterOrtExtensions = false;
                LoggingLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING;
            }
            
            [UnscopedRef]
            public ref ConfigBuilder WithBackendType(BackendType backendType, int deviceID = 0)
            {
                BackendType = backendType;
                DeviceID = deviceID;
                
                return ref this;
            }
            
            [UnscopedRef]
            public ref ConfigBuilder WithMemoryMode(OnnxMemoryModes memoryMode)
            {
                MemoryMode = memoryMode;
                
                return ref this;
            }
            
            [UnscopedRef]
            public ref ConfigBuilder WithRegisterOrtExtensions()
            {
                RegisterOrtExtensions = true;

                return ref this;
            }
            
            [UnscopedRef]
            public ref ConfigBuilder WithLoggingLevel(OrtLoggingLevel loggingLevel)
            {
                LoggingLevel = loggingLevel;

                return ref this;
            }
            
            internal BuiltConfig Build()
            {
                return new(this);
            }
        }
    
        public interface IConfig
        {
            public static abstract ConfigBuilder ConfigBuilder { get; }
        }
    }
    
    public partial struct ConfigurableOnnxModel<ConfigT>: IDisposable
        where ConfigT: struct, ConfigurableOnnxModel.IConfig
    {
        private static ConfigurableOnnxModel.BuiltConfig CONFIG => BuiltConfigCache<ConfigT>.BUILT_CONFIG;
        
        public ConfigurableOnnxModel.BuiltConfig Config => CONFIG;

        [NoParamlessCtor]
        public readonly partial struct SessionHandle: IDisposable
        {
            public readonly InferenceSession Session;
            
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            internal SessionHandle(InferenceSession session)
            {
                Session = session;
            }
            
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public void Dispose()
            {
                if (CONFIG.MemoryMode.HasFlag(OnnxMemoryModes.UnloadAfterUse))
                {
                    Session?.Dispose();
                }
            }
        }

        private readonly string ModelPath;

        private InferenceSession? Session;

        [Obsolete("Use constructor with parameters.", error: true)]
        public ConfigurableOnnxModel() { }

        public ConfigurableOnnxModel(string modelPath)
        {
            ModelPath = modelPath;

            var memoryMode = Config.MemoryMode;
            
            if (!memoryMode.HasFlag(OnnxMemoryModes.DeferLoading) &&
                !memoryMode.HasFlag(OnnxMemoryModes.UnloadAfterUse))
            {
                Session = CreateSession(modelPath);
            }
        }
        
        // It looks deceptively bloated, but the branches are optimized away
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public SessionHandle GetSessionHandle()
        {
            var memoryMode = Config.MemoryMode;

            var modelPath = ModelPath;

            InferenceSession session;
            
            // If UnloadAfterUse is set, we can optimize away the null check,
            // since we know that a new session will always be created.
            // UnloadAfterUse is also implicitly DeferLoading.
            if (memoryMode.HasFlag(OnnxMemoryModes.UnloadAfterUse))
            {
                session = CreateSession(modelPath);
            }
            
            // If we are deferring loading, we still cache the model...
            else if (memoryMode.HasFlag(OnnxMemoryModes.DeferLoading))
            {
                session = (Session ??= CreateSession(modelPath));
            }

            else //The model is already cached!
            {
                session = Session!;
            }
            
            return new(session);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static InferenceSession CreateSession(string modelPath)
        {
            var config = CONFIG;
            
            return new(modelPath, options: config.SessionOptions);
        }
        
        public void Dispose()
        {
            Session?.Dispose();
        }
    }

    internal static class BuiltConfigCache<ConfigT> where ConfigT: struct, ConfigurableOnnxModel.IConfig
    {
        public static readonly ConfigurableOnnxModel.BuiltConfig BUILT_CONFIG = ConfigT.ConfigBuilder.Build();
    }
}