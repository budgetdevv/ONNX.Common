using System;
using System.Diagnostics.CodeAnalysis;
using Microsoft.ML.OnnxRuntime;
using ONNX.Common.Configs;

namespace ONNX.Common
{
    public readonly struct ConfigurableOnnxModel: IDisposable
    {
        public struct Configuration
        {
            public SessionOptions SessionOptions;
            
            public string? ModelPath;  
            
            public BackendType BackendType;

            public int DeviceID;
            
            public OnnxMemoryModes MemoryMode;
            
            public bool RegisterOrtExtensions;
            
            public OrtLoggingLevel LoggingLevel;

            public Configuration()
            {
                SessionOptions = new();
                ModelPath = null;
                BackendType = BackendType.CPU;
                DeviceID = 0;
                MemoryMode = OnnxMemoryModes.None;
                RegisterOrtExtensions = false;
                LoggingLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING;
            }

            [UnscopedRef]
            public ref Configuration WithModelPath(string? modelPath)
            {
                ModelPath = modelPath;
                
                return ref this;
            }
            
            [UnscopedRef]
            public ref Configuration WithBackendType(BackendType backendType, int deviceID = 0)
            {
                BackendType = backendType;
                DeviceID = deviceID;
                
                return ref this;
            }
            
            [UnscopedRef]
            public ref Configuration WithMemoryMode(OnnxMemoryModes memoryMode)
            {
                MemoryMode = memoryMode;
                
                return ref this;
            }
            
            [UnscopedRef]
            public ref Configuration WithRegisterOrtExtensions()
            {
                RegisterOrtExtensions = true;

                return ref this;
            }
            
            [UnscopedRef]
            public ref Configuration WithLoggingLevel(OrtLoggingLevel loggingLevel)
            {
                LoggingLevel = loggingLevel;

                return ref this;
            }

            public ConfigurableOnnxModel CreateModel()
            {
                return new(this);
            }
        }
        
        public readonly InferenceSession Session;

        private readonly SessionOptions SessionOptions;

        private readonly Configuration Config;
        
        public ConfigurableOnnxModel()
        {
            throw new NotSupportedException();
        }
        
        public ConfigurableOnnxModel(Configuration config)
        {
            Config = config;
            
            var sessionOptions = SessionOptions = config.SessionOptions;
            
            if (config.RegisterOrtExtensions)
            {
                sessionOptions.RegisterOrtExtensions();
            }
            
            var deviceID = config.DeviceID;

            switch (config.BackendType)
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

            sessionOptions.LogSeverityLevel = config.LoggingLevel;
            
            Session = new(modelPath: config.ModelPath, options: sessionOptions);
        }

        public void Dispose()
        {
            Session.Dispose();
            SessionOptions.Dispose();
        }
    }
}