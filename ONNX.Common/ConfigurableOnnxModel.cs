using System;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using Microsoft.ML.OnnxRuntime;
using ONNX.Common.Configs;

namespace ONNX.Common
{
    public static class ConfigurableOnnxModel
    {
        public struct BuiltConfig
        {
            public SessionOptions SessionOptions;
            
            public string ModelPath;  
            
            public BackendType BackendType;

            public int DeviceID;
            
            public OnnxMemoryModes MemoryMode;
            
            public bool RegisterOrtExtensions;
            
            public OrtLoggingLevel LoggingLevel;
            
            [Obsolete("Use constructor with parameters", error: true)]
            public BuiltConfig()
            {
                throw new NotSupportedException();
            }
            
            public BuiltConfig(ConfigBuilder configBuilder)
            {
                var sessionOptions = SessionOptions = new();
                
                ModelPath = configBuilder.ModelPath ?? throw new ArgumentNullException(nameof(configBuilder.ModelPath));
                
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
            public string? ModelPath;  
            
            public BackendType BackendType;

            public int DeviceID;
            
            public OnnxMemoryModes MemoryMode;
            
            public bool RegisterOrtExtensions;
            
            public OrtLoggingLevel LoggingLevel;

            public ConfigBuilder()
            {
                ModelPath = null;
                BackendType = BackendType.CPU;
                DeviceID = 0;
                MemoryMode = OnnxMemoryModes.None;
                RegisterOrtExtensions = false;
                LoggingLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING;
            }

            [UnscopedRef]
            public ref ConfigBuilder WithModelPath(string? modelPath)
            {
                ModelPath = modelPath;
                
                return ref this;
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
            
            public BuiltConfig Build()
            {
                return new(this);
            }
        }
    
        public interface IConfig
        {
            public static abstract BuiltConfig Config { get; }
        }
    }
    
    public struct ConfigurableOnnxModel<ConfigT>: IDisposable
        where ConfigT: struct, ConfigurableOnnxModel.IConfig
    {
        public readonly struct SessionHandle: IDisposable
        {
            public readonly InferenceSession Session;

            [Obsolete("Use constructor with parameters", error: true)]
            public SessionHandle()
            {
                throw new NotSupportedException();
            }
            
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            internal SessionHandle(InferenceSession session)
            {
                Session = session;
            }
            
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public void Dispose()
            {
                if (ConfigT.Config.MemoryMode.HasFlag(OnnxMemoryModes.UnloadAfterUse))
                {
                    Session?.Dispose();
                }
            }
        }
        
        private InferenceSession? Session;
        
        public ConfigurableOnnxModel()
        {
            var memoryMode = ConfigT.Config.MemoryMode;
            
            if (!memoryMode.HasFlag(OnnxMemoryModes.DeferLoading) &&
                !memoryMode.HasFlag(OnnxMemoryModes.UnloadAfterUse))
            {
                Session = CreateSession();
            }
        }
        
        // It looks deceptively bloated, but the branches are optimized away
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public SessionHandle GetSessionHandle()
        {
            var memoryMode = ConfigT.Config.MemoryMode;

            InferenceSession session;
            
            // If UnloadAfterUse is set, we can optimize away the null check,
            // since we know that a new session will always be created.
            // UnloadAfterUse is also implicitly DeferLoading.
            if (memoryMode.HasFlag(OnnxMemoryModes.UnloadAfterUse))
            {
                session = CreateSession();
            }
            
            // If we are deferring loading, we still cache the model...
            else if (memoryMode.HasFlag(OnnxMemoryModes.DeferLoading))
            {
                session = (Session ??= CreateSession());
            }

            else //The model is already cached!
            {
                session = Session!;
            }
            
            return new(session);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static InferenceSession CreateSession()
        {
            var config = ConfigT.Config;
            
            return new(modelPath: config.ModelPath, options: config.SessionOptions);
        }
        
        public void Dispose()
        {
            Session?.Dispose();
        }
    }
}