using System;
using System.Diagnostics.CodeAnalysis;
using FlorenceSharp.Configs;
using Microsoft.ML.OnnxRuntime;

namespace FlorenceSharp
{
    public readonly struct ConfigurableOnnxModel: IDisposable
    {
        public struct Configuration
        {
            public SessionOptions SessionOptions;
            
            public string? ModelPath;  
            
            public DeviceType DeviceType;

            public int DeviceID;
            
            public OnnxMemoryModes MemoryMode;
            
            public bool RegisterOrtExtensions;
            
            public OrtLoggingLevel LoggingLevel;

            public Configuration()
            {
                SessionOptions = new();
                ModelPath = null;
                DeviceType = DeviceType.CPU;
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
            public ref Configuration WithDeviceType(DeviceType deviceType, int deviceID = 0)
            {
                DeviceType = deviceType;
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
        
        public ConfigurableOnnxModel(Configuration config)
        {
            Config = config;
            
            var sessionOptions = SessionOptions = config.SessionOptions;
            
            if (config.RegisterOrtExtensions)
            {
                sessionOptions.RegisterOrtExtensions();
            }
            
            var deviceID = config.DeviceID;

            switch (config.DeviceType)
            {
                // "Unhandled exception. Microsoft.ML.OnnxRuntime.OnnxRuntimeException: [ErrorCode:Fail] Provider CPUExecutionProvider has already been registered."
                // case DeviceType.CPU:
                //     sessionOptions.AppendExecutionProvider_CPU();
                //     break;
                
                case DeviceType.CUDA:
                    sessionOptions.AppendExecutionProvider_CUDA(deviceID);
                    break;
                
                case DeviceType.DirectML:
                    sessionOptions.AppendExecutionProvider_DML(deviceID);
                    break;
                
                case DeviceType.CoreML:
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