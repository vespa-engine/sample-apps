package inference;

import static io.grpc.MethodDescriptor.generateFullMethodName;

/**
 * <pre>
 *&#64;&#64;
 *&#64;&#64;.. cpp:var:: service InferenceService
 *&#64;&#64;
 *&#64;&#64;   Inference Server GRPC endpoints.
 *&#64;&#64;
 * </pre>
 */
@javax.annotation.Generated(
    value = "by gRPC proto compiler (version 1.59.1)",
    comments = "Source: grpc_service.proto")
@io.grpc.stub.annotations.GrpcGenerated
public final class GRPCInferenceServiceGrpc {

  private GRPCInferenceServiceGrpc() {}

  public static final java.lang.String SERVICE_NAME = "inference.GRPCInferenceService";

  // Static method descriptors that strictly reflect the proto.
  private static volatile io.grpc.MethodDescriptor<inference.GrpcService.ServerLiveRequest,
      inference.GrpcService.ServerLiveResponse> getServerLiveMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "ServerLive",
      requestType = inference.GrpcService.ServerLiveRequest.class,
      responseType = inference.GrpcService.ServerLiveResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<inference.GrpcService.ServerLiveRequest,
      inference.GrpcService.ServerLiveResponse> getServerLiveMethod() {
    io.grpc.MethodDescriptor<inference.GrpcService.ServerLiveRequest, inference.GrpcService.ServerLiveResponse> getServerLiveMethod;
    if ((getServerLiveMethod = GRPCInferenceServiceGrpc.getServerLiveMethod) == null) {
      synchronized (GRPCInferenceServiceGrpc.class) {
        if ((getServerLiveMethod = GRPCInferenceServiceGrpc.getServerLiveMethod) == null) {
          GRPCInferenceServiceGrpc.getServerLiveMethod = getServerLiveMethod =
              io.grpc.MethodDescriptor.<inference.GrpcService.ServerLiveRequest, inference.GrpcService.ServerLiveResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "ServerLive"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  inference.GrpcService.ServerLiveRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  inference.GrpcService.ServerLiveResponse.getDefaultInstance()))
              .setSchemaDescriptor(new GRPCInferenceServiceMethodDescriptorSupplier("ServerLive"))
              .build();
        }
      }
    }
    return getServerLiveMethod;
  }

  private static volatile io.grpc.MethodDescriptor<inference.GrpcService.ServerReadyRequest,
      inference.GrpcService.ServerReadyResponse> getServerReadyMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "ServerReady",
      requestType = inference.GrpcService.ServerReadyRequest.class,
      responseType = inference.GrpcService.ServerReadyResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<inference.GrpcService.ServerReadyRequest,
      inference.GrpcService.ServerReadyResponse> getServerReadyMethod() {
    io.grpc.MethodDescriptor<inference.GrpcService.ServerReadyRequest, inference.GrpcService.ServerReadyResponse> getServerReadyMethod;
    if ((getServerReadyMethod = GRPCInferenceServiceGrpc.getServerReadyMethod) == null) {
      synchronized (GRPCInferenceServiceGrpc.class) {
        if ((getServerReadyMethod = GRPCInferenceServiceGrpc.getServerReadyMethod) == null) {
          GRPCInferenceServiceGrpc.getServerReadyMethod = getServerReadyMethod =
              io.grpc.MethodDescriptor.<inference.GrpcService.ServerReadyRequest, inference.GrpcService.ServerReadyResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "ServerReady"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  inference.GrpcService.ServerReadyRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  inference.GrpcService.ServerReadyResponse.getDefaultInstance()))
              .setSchemaDescriptor(new GRPCInferenceServiceMethodDescriptorSupplier("ServerReady"))
              .build();
        }
      }
    }
    return getServerReadyMethod;
  }

  private static volatile io.grpc.MethodDescriptor<inference.GrpcService.ModelReadyRequest,
      inference.GrpcService.ModelReadyResponse> getModelReadyMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "ModelReady",
      requestType = inference.GrpcService.ModelReadyRequest.class,
      responseType = inference.GrpcService.ModelReadyResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<inference.GrpcService.ModelReadyRequest,
      inference.GrpcService.ModelReadyResponse> getModelReadyMethod() {
    io.grpc.MethodDescriptor<inference.GrpcService.ModelReadyRequest, inference.GrpcService.ModelReadyResponse> getModelReadyMethod;
    if ((getModelReadyMethod = GRPCInferenceServiceGrpc.getModelReadyMethod) == null) {
      synchronized (GRPCInferenceServiceGrpc.class) {
        if ((getModelReadyMethod = GRPCInferenceServiceGrpc.getModelReadyMethod) == null) {
          GRPCInferenceServiceGrpc.getModelReadyMethod = getModelReadyMethod =
              io.grpc.MethodDescriptor.<inference.GrpcService.ModelReadyRequest, inference.GrpcService.ModelReadyResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "ModelReady"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  inference.GrpcService.ModelReadyRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  inference.GrpcService.ModelReadyResponse.getDefaultInstance()))
              .setSchemaDescriptor(new GRPCInferenceServiceMethodDescriptorSupplier("ModelReady"))
              .build();
        }
      }
    }
    return getModelReadyMethod;
  }

  private static volatile io.grpc.MethodDescriptor<inference.GrpcService.ServerMetadataRequest,
      inference.GrpcService.ServerMetadataResponse> getServerMetadataMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "ServerMetadata",
      requestType = inference.GrpcService.ServerMetadataRequest.class,
      responseType = inference.GrpcService.ServerMetadataResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<inference.GrpcService.ServerMetadataRequest,
      inference.GrpcService.ServerMetadataResponse> getServerMetadataMethod() {
    io.grpc.MethodDescriptor<inference.GrpcService.ServerMetadataRequest, inference.GrpcService.ServerMetadataResponse> getServerMetadataMethod;
    if ((getServerMetadataMethod = GRPCInferenceServiceGrpc.getServerMetadataMethod) == null) {
      synchronized (GRPCInferenceServiceGrpc.class) {
        if ((getServerMetadataMethod = GRPCInferenceServiceGrpc.getServerMetadataMethod) == null) {
          GRPCInferenceServiceGrpc.getServerMetadataMethod = getServerMetadataMethod =
              io.grpc.MethodDescriptor.<inference.GrpcService.ServerMetadataRequest, inference.GrpcService.ServerMetadataResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "ServerMetadata"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  inference.GrpcService.ServerMetadataRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  inference.GrpcService.ServerMetadataResponse.getDefaultInstance()))
              .setSchemaDescriptor(new GRPCInferenceServiceMethodDescriptorSupplier("ServerMetadata"))
              .build();
        }
      }
    }
    return getServerMetadataMethod;
  }

  private static volatile io.grpc.MethodDescriptor<inference.GrpcService.ModelMetadataRequest,
      inference.GrpcService.ModelMetadataResponse> getModelMetadataMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "ModelMetadata",
      requestType = inference.GrpcService.ModelMetadataRequest.class,
      responseType = inference.GrpcService.ModelMetadataResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<inference.GrpcService.ModelMetadataRequest,
      inference.GrpcService.ModelMetadataResponse> getModelMetadataMethod() {
    io.grpc.MethodDescriptor<inference.GrpcService.ModelMetadataRequest, inference.GrpcService.ModelMetadataResponse> getModelMetadataMethod;
    if ((getModelMetadataMethod = GRPCInferenceServiceGrpc.getModelMetadataMethod) == null) {
      synchronized (GRPCInferenceServiceGrpc.class) {
        if ((getModelMetadataMethod = GRPCInferenceServiceGrpc.getModelMetadataMethod) == null) {
          GRPCInferenceServiceGrpc.getModelMetadataMethod = getModelMetadataMethod =
              io.grpc.MethodDescriptor.<inference.GrpcService.ModelMetadataRequest, inference.GrpcService.ModelMetadataResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "ModelMetadata"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  inference.GrpcService.ModelMetadataRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  inference.GrpcService.ModelMetadataResponse.getDefaultInstance()))
              .setSchemaDescriptor(new GRPCInferenceServiceMethodDescriptorSupplier("ModelMetadata"))
              .build();
        }
      }
    }
    return getModelMetadataMethod;
  }

  private static volatile io.grpc.MethodDescriptor<inference.GrpcService.ModelInferRequest,
      inference.GrpcService.ModelInferResponse> getModelInferMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "ModelInfer",
      requestType = inference.GrpcService.ModelInferRequest.class,
      responseType = inference.GrpcService.ModelInferResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<inference.GrpcService.ModelInferRequest,
      inference.GrpcService.ModelInferResponse> getModelInferMethod() {
    io.grpc.MethodDescriptor<inference.GrpcService.ModelInferRequest, inference.GrpcService.ModelInferResponse> getModelInferMethod;
    if ((getModelInferMethod = GRPCInferenceServiceGrpc.getModelInferMethod) == null) {
      synchronized (GRPCInferenceServiceGrpc.class) {
        if ((getModelInferMethod = GRPCInferenceServiceGrpc.getModelInferMethod) == null) {
          GRPCInferenceServiceGrpc.getModelInferMethod = getModelInferMethod =
              io.grpc.MethodDescriptor.<inference.GrpcService.ModelInferRequest, inference.GrpcService.ModelInferResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "ModelInfer"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  inference.GrpcService.ModelInferRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  inference.GrpcService.ModelInferResponse.getDefaultInstance()))
              .setSchemaDescriptor(new GRPCInferenceServiceMethodDescriptorSupplier("ModelInfer"))
              .build();
        }
      }
    }
    return getModelInferMethod;
  }

  private static volatile io.grpc.MethodDescriptor<inference.GrpcService.ModelInferRequest,
      inference.GrpcService.ModelStreamInferResponse> getModelStreamInferMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "ModelStreamInfer",
      requestType = inference.GrpcService.ModelInferRequest.class,
      responseType = inference.GrpcService.ModelStreamInferResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.BIDI_STREAMING)
  public static io.grpc.MethodDescriptor<inference.GrpcService.ModelInferRequest,
      inference.GrpcService.ModelStreamInferResponse> getModelStreamInferMethod() {
    io.grpc.MethodDescriptor<inference.GrpcService.ModelInferRequest, inference.GrpcService.ModelStreamInferResponse> getModelStreamInferMethod;
    if ((getModelStreamInferMethod = GRPCInferenceServiceGrpc.getModelStreamInferMethod) == null) {
      synchronized (GRPCInferenceServiceGrpc.class) {
        if ((getModelStreamInferMethod = GRPCInferenceServiceGrpc.getModelStreamInferMethod) == null) {
          GRPCInferenceServiceGrpc.getModelStreamInferMethod = getModelStreamInferMethod =
              io.grpc.MethodDescriptor.<inference.GrpcService.ModelInferRequest, inference.GrpcService.ModelStreamInferResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.BIDI_STREAMING)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "ModelStreamInfer"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  inference.GrpcService.ModelInferRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  inference.GrpcService.ModelStreamInferResponse.getDefaultInstance()))
              .setSchemaDescriptor(new GRPCInferenceServiceMethodDescriptorSupplier("ModelStreamInfer"))
              .build();
        }
      }
    }
    return getModelStreamInferMethod;
  }

  private static volatile io.grpc.MethodDescriptor<inference.GrpcService.ModelConfigRequest,
      inference.GrpcService.ModelConfigResponse> getModelConfigMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "ModelConfig",
      requestType = inference.GrpcService.ModelConfigRequest.class,
      responseType = inference.GrpcService.ModelConfigResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<inference.GrpcService.ModelConfigRequest,
      inference.GrpcService.ModelConfigResponse> getModelConfigMethod() {
    io.grpc.MethodDescriptor<inference.GrpcService.ModelConfigRequest, inference.GrpcService.ModelConfigResponse> getModelConfigMethod;
    if ((getModelConfigMethod = GRPCInferenceServiceGrpc.getModelConfigMethod) == null) {
      synchronized (GRPCInferenceServiceGrpc.class) {
        if ((getModelConfigMethod = GRPCInferenceServiceGrpc.getModelConfigMethod) == null) {
          GRPCInferenceServiceGrpc.getModelConfigMethod = getModelConfigMethod =
              io.grpc.MethodDescriptor.<inference.GrpcService.ModelConfigRequest, inference.GrpcService.ModelConfigResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "ModelConfig"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  inference.GrpcService.ModelConfigRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  inference.GrpcService.ModelConfigResponse.getDefaultInstance()))
              .setSchemaDescriptor(new GRPCInferenceServiceMethodDescriptorSupplier("ModelConfig"))
              .build();
        }
      }
    }
    return getModelConfigMethod;
  }

  private static volatile io.grpc.MethodDescriptor<inference.GrpcService.ModelStatisticsRequest,
      inference.GrpcService.ModelStatisticsResponse> getModelStatisticsMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "ModelStatistics",
      requestType = inference.GrpcService.ModelStatisticsRequest.class,
      responseType = inference.GrpcService.ModelStatisticsResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<inference.GrpcService.ModelStatisticsRequest,
      inference.GrpcService.ModelStatisticsResponse> getModelStatisticsMethod() {
    io.grpc.MethodDescriptor<inference.GrpcService.ModelStatisticsRequest, inference.GrpcService.ModelStatisticsResponse> getModelStatisticsMethod;
    if ((getModelStatisticsMethod = GRPCInferenceServiceGrpc.getModelStatisticsMethod) == null) {
      synchronized (GRPCInferenceServiceGrpc.class) {
        if ((getModelStatisticsMethod = GRPCInferenceServiceGrpc.getModelStatisticsMethod) == null) {
          GRPCInferenceServiceGrpc.getModelStatisticsMethod = getModelStatisticsMethod =
              io.grpc.MethodDescriptor.<inference.GrpcService.ModelStatisticsRequest, inference.GrpcService.ModelStatisticsResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "ModelStatistics"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  inference.GrpcService.ModelStatisticsRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  inference.GrpcService.ModelStatisticsResponse.getDefaultInstance()))
              .setSchemaDescriptor(new GRPCInferenceServiceMethodDescriptorSupplier("ModelStatistics"))
              .build();
        }
      }
    }
    return getModelStatisticsMethod;
  }

  private static volatile io.grpc.MethodDescriptor<inference.GrpcService.RepositoryIndexRequest,
      inference.GrpcService.RepositoryIndexResponse> getRepositoryIndexMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "RepositoryIndex",
      requestType = inference.GrpcService.RepositoryIndexRequest.class,
      responseType = inference.GrpcService.RepositoryIndexResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<inference.GrpcService.RepositoryIndexRequest,
      inference.GrpcService.RepositoryIndexResponse> getRepositoryIndexMethod() {
    io.grpc.MethodDescriptor<inference.GrpcService.RepositoryIndexRequest, inference.GrpcService.RepositoryIndexResponse> getRepositoryIndexMethod;
    if ((getRepositoryIndexMethod = GRPCInferenceServiceGrpc.getRepositoryIndexMethod) == null) {
      synchronized (GRPCInferenceServiceGrpc.class) {
        if ((getRepositoryIndexMethod = GRPCInferenceServiceGrpc.getRepositoryIndexMethod) == null) {
          GRPCInferenceServiceGrpc.getRepositoryIndexMethod = getRepositoryIndexMethod =
              io.grpc.MethodDescriptor.<inference.GrpcService.RepositoryIndexRequest, inference.GrpcService.RepositoryIndexResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "RepositoryIndex"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  inference.GrpcService.RepositoryIndexRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  inference.GrpcService.RepositoryIndexResponse.getDefaultInstance()))
              .setSchemaDescriptor(new GRPCInferenceServiceMethodDescriptorSupplier("RepositoryIndex"))
              .build();
        }
      }
    }
    return getRepositoryIndexMethod;
  }

  private static volatile io.grpc.MethodDescriptor<inference.GrpcService.RepositoryModelLoadRequest,
      inference.GrpcService.RepositoryModelLoadResponse> getRepositoryModelLoadMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "RepositoryModelLoad",
      requestType = inference.GrpcService.RepositoryModelLoadRequest.class,
      responseType = inference.GrpcService.RepositoryModelLoadResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<inference.GrpcService.RepositoryModelLoadRequest,
      inference.GrpcService.RepositoryModelLoadResponse> getRepositoryModelLoadMethod() {
    io.grpc.MethodDescriptor<inference.GrpcService.RepositoryModelLoadRequest, inference.GrpcService.RepositoryModelLoadResponse> getRepositoryModelLoadMethod;
    if ((getRepositoryModelLoadMethod = GRPCInferenceServiceGrpc.getRepositoryModelLoadMethod) == null) {
      synchronized (GRPCInferenceServiceGrpc.class) {
        if ((getRepositoryModelLoadMethod = GRPCInferenceServiceGrpc.getRepositoryModelLoadMethod) == null) {
          GRPCInferenceServiceGrpc.getRepositoryModelLoadMethod = getRepositoryModelLoadMethod =
              io.grpc.MethodDescriptor.<inference.GrpcService.RepositoryModelLoadRequest, inference.GrpcService.RepositoryModelLoadResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "RepositoryModelLoad"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  inference.GrpcService.RepositoryModelLoadRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  inference.GrpcService.RepositoryModelLoadResponse.getDefaultInstance()))
              .setSchemaDescriptor(new GRPCInferenceServiceMethodDescriptorSupplier("RepositoryModelLoad"))
              .build();
        }
      }
    }
    return getRepositoryModelLoadMethod;
  }

  private static volatile io.grpc.MethodDescriptor<inference.GrpcService.RepositoryModelUnloadRequest,
      inference.GrpcService.RepositoryModelUnloadResponse> getRepositoryModelUnloadMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "RepositoryModelUnload",
      requestType = inference.GrpcService.RepositoryModelUnloadRequest.class,
      responseType = inference.GrpcService.RepositoryModelUnloadResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<inference.GrpcService.RepositoryModelUnloadRequest,
      inference.GrpcService.RepositoryModelUnloadResponse> getRepositoryModelUnloadMethod() {
    io.grpc.MethodDescriptor<inference.GrpcService.RepositoryModelUnloadRequest, inference.GrpcService.RepositoryModelUnloadResponse> getRepositoryModelUnloadMethod;
    if ((getRepositoryModelUnloadMethod = GRPCInferenceServiceGrpc.getRepositoryModelUnloadMethod) == null) {
      synchronized (GRPCInferenceServiceGrpc.class) {
        if ((getRepositoryModelUnloadMethod = GRPCInferenceServiceGrpc.getRepositoryModelUnloadMethod) == null) {
          GRPCInferenceServiceGrpc.getRepositoryModelUnloadMethod = getRepositoryModelUnloadMethod =
              io.grpc.MethodDescriptor.<inference.GrpcService.RepositoryModelUnloadRequest, inference.GrpcService.RepositoryModelUnloadResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "RepositoryModelUnload"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  inference.GrpcService.RepositoryModelUnloadRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  inference.GrpcService.RepositoryModelUnloadResponse.getDefaultInstance()))
              .setSchemaDescriptor(new GRPCInferenceServiceMethodDescriptorSupplier("RepositoryModelUnload"))
              .build();
        }
      }
    }
    return getRepositoryModelUnloadMethod;
  }

  private static volatile io.grpc.MethodDescriptor<inference.GrpcService.SystemSharedMemoryStatusRequest,
      inference.GrpcService.SystemSharedMemoryStatusResponse> getSystemSharedMemoryStatusMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "SystemSharedMemoryStatus",
      requestType = inference.GrpcService.SystemSharedMemoryStatusRequest.class,
      responseType = inference.GrpcService.SystemSharedMemoryStatusResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<inference.GrpcService.SystemSharedMemoryStatusRequest,
      inference.GrpcService.SystemSharedMemoryStatusResponse> getSystemSharedMemoryStatusMethod() {
    io.grpc.MethodDescriptor<inference.GrpcService.SystemSharedMemoryStatusRequest, inference.GrpcService.SystemSharedMemoryStatusResponse> getSystemSharedMemoryStatusMethod;
    if ((getSystemSharedMemoryStatusMethod = GRPCInferenceServiceGrpc.getSystemSharedMemoryStatusMethod) == null) {
      synchronized (GRPCInferenceServiceGrpc.class) {
        if ((getSystemSharedMemoryStatusMethod = GRPCInferenceServiceGrpc.getSystemSharedMemoryStatusMethod) == null) {
          GRPCInferenceServiceGrpc.getSystemSharedMemoryStatusMethod = getSystemSharedMemoryStatusMethod =
              io.grpc.MethodDescriptor.<inference.GrpcService.SystemSharedMemoryStatusRequest, inference.GrpcService.SystemSharedMemoryStatusResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "SystemSharedMemoryStatus"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  inference.GrpcService.SystemSharedMemoryStatusRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  inference.GrpcService.SystemSharedMemoryStatusResponse.getDefaultInstance()))
              .setSchemaDescriptor(new GRPCInferenceServiceMethodDescriptorSupplier("SystemSharedMemoryStatus"))
              .build();
        }
      }
    }
    return getSystemSharedMemoryStatusMethod;
  }

  private static volatile io.grpc.MethodDescriptor<inference.GrpcService.SystemSharedMemoryRegisterRequest,
      inference.GrpcService.SystemSharedMemoryRegisterResponse> getSystemSharedMemoryRegisterMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "SystemSharedMemoryRegister",
      requestType = inference.GrpcService.SystemSharedMemoryRegisterRequest.class,
      responseType = inference.GrpcService.SystemSharedMemoryRegisterResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<inference.GrpcService.SystemSharedMemoryRegisterRequest,
      inference.GrpcService.SystemSharedMemoryRegisterResponse> getSystemSharedMemoryRegisterMethod() {
    io.grpc.MethodDescriptor<inference.GrpcService.SystemSharedMemoryRegisterRequest, inference.GrpcService.SystemSharedMemoryRegisterResponse> getSystemSharedMemoryRegisterMethod;
    if ((getSystemSharedMemoryRegisterMethod = GRPCInferenceServiceGrpc.getSystemSharedMemoryRegisterMethod) == null) {
      synchronized (GRPCInferenceServiceGrpc.class) {
        if ((getSystemSharedMemoryRegisterMethod = GRPCInferenceServiceGrpc.getSystemSharedMemoryRegisterMethod) == null) {
          GRPCInferenceServiceGrpc.getSystemSharedMemoryRegisterMethod = getSystemSharedMemoryRegisterMethod =
              io.grpc.MethodDescriptor.<inference.GrpcService.SystemSharedMemoryRegisterRequest, inference.GrpcService.SystemSharedMemoryRegisterResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "SystemSharedMemoryRegister"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  inference.GrpcService.SystemSharedMemoryRegisterRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  inference.GrpcService.SystemSharedMemoryRegisterResponse.getDefaultInstance()))
              .setSchemaDescriptor(new GRPCInferenceServiceMethodDescriptorSupplier("SystemSharedMemoryRegister"))
              .build();
        }
      }
    }
    return getSystemSharedMemoryRegisterMethod;
  }

  private static volatile io.grpc.MethodDescriptor<inference.GrpcService.SystemSharedMemoryUnregisterRequest,
      inference.GrpcService.SystemSharedMemoryUnregisterResponse> getSystemSharedMemoryUnregisterMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "SystemSharedMemoryUnregister",
      requestType = inference.GrpcService.SystemSharedMemoryUnregisterRequest.class,
      responseType = inference.GrpcService.SystemSharedMemoryUnregisterResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<inference.GrpcService.SystemSharedMemoryUnregisterRequest,
      inference.GrpcService.SystemSharedMemoryUnregisterResponse> getSystemSharedMemoryUnregisterMethod() {
    io.grpc.MethodDescriptor<inference.GrpcService.SystemSharedMemoryUnregisterRequest, inference.GrpcService.SystemSharedMemoryUnregisterResponse> getSystemSharedMemoryUnregisterMethod;
    if ((getSystemSharedMemoryUnregisterMethod = GRPCInferenceServiceGrpc.getSystemSharedMemoryUnregisterMethod) == null) {
      synchronized (GRPCInferenceServiceGrpc.class) {
        if ((getSystemSharedMemoryUnregisterMethod = GRPCInferenceServiceGrpc.getSystemSharedMemoryUnregisterMethod) == null) {
          GRPCInferenceServiceGrpc.getSystemSharedMemoryUnregisterMethod = getSystemSharedMemoryUnregisterMethod =
              io.grpc.MethodDescriptor.<inference.GrpcService.SystemSharedMemoryUnregisterRequest, inference.GrpcService.SystemSharedMemoryUnregisterResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "SystemSharedMemoryUnregister"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  inference.GrpcService.SystemSharedMemoryUnregisterRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  inference.GrpcService.SystemSharedMemoryUnregisterResponse.getDefaultInstance()))
              .setSchemaDescriptor(new GRPCInferenceServiceMethodDescriptorSupplier("SystemSharedMemoryUnregister"))
              .build();
        }
      }
    }
    return getSystemSharedMemoryUnregisterMethod;
  }

  private static volatile io.grpc.MethodDescriptor<inference.GrpcService.CudaSharedMemoryStatusRequest,
      inference.GrpcService.CudaSharedMemoryStatusResponse> getCudaSharedMemoryStatusMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "CudaSharedMemoryStatus",
      requestType = inference.GrpcService.CudaSharedMemoryStatusRequest.class,
      responseType = inference.GrpcService.CudaSharedMemoryStatusResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<inference.GrpcService.CudaSharedMemoryStatusRequest,
      inference.GrpcService.CudaSharedMemoryStatusResponse> getCudaSharedMemoryStatusMethod() {
    io.grpc.MethodDescriptor<inference.GrpcService.CudaSharedMemoryStatusRequest, inference.GrpcService.CudaSharedMemoryStatusResponse> getCudaSharedMemoryStatusMethod;
    if ((getCudaSharedMemoryStatusMethod = GRPCInferenceServiceGrpc.getCudaSharedMemoryStatusMethod) == null) {
      synchronized (GRPCInferenceServiceGrpc.class) {
        if ((getCudaSharedMemoryStatusMethod = GRPCInferenceServiceGrpc.getCudaSharedMemoryStatusMethod) == null) {
          GRPCInferenceServiceGrpc.getCudaSharedMemoryStatusMethod = getCudaSharedMemoryStatusMethod =
              io.grpc.MethodDescriptor.<inference.GrpcService.CudaSharedMemoryStatusRequest, inference.GrpcService.CudaSharedMemoryStatusResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "CudaSharedMemoryStatus"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  inference.GrpcService.CudaSharedMemoryStatusRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  inference.GrpcService.CudaSharedMemoryStatusResponse.getDefaultInstance()))
              .setSchemaDescriptor(new GRPCInferenceServiceMethodDescriptorSupplier("CudaSharedMemoryStatus"))
              .build();
        }
      }
    }
    return getCudaSharedMemoryStatusMethod;
  }

  private static volatile io.grpc.MethodDescriptor<inference.GrpcService.CudaSharedMemoryRegisterRequest,
      inference.GrpcService.CudaSharedMemoryRegisterResponse> getCudaSharedMemoryRegisterMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "CudaSharedMemoryRegister",
      requestType = inference.GrpcService.CudaSharedMemoryRegisterRequest.class,
      responseType = inference.GrpcService.CudaSharedMemoryRegisterResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<inference.GrpcService.CudaSharedMemoryRegisterRequest,
      inference.GrpcService.CudaSharedMemoryRegisterResponse> getCudaSharedMemoryRegisterMethod() {
    io.grpc.MethodDescriptor<inference.GrpcService.CudaSharedMemoryRegisterRequest, inference.GrpcService.CudaSharedMemoryRegisterResponse> getCudaSharedMemoryRegisterMethod;
    if ((getCudaSharedMemoryRegisterMethod = GRPCInferenceServiceGrpc.getCudaSharedMemoryRegisterMethod) == null) {
      synchronized (GRPCInferenceServiceGrpc.class) {
        if ((getCudaSharedMemoryRegisterMethod = GRPCInferenceServiceGrpc.getCudaSharedMemoryRegisterMethod) == null) {
          GRPCInferenceServiceGrpc.getCudaSharedMemoryRegisterMethod = getCudaSharedMemoryRegisterMethod =
              io.grpc.MethodDescriptor.<inference.GrpcService.CudaSharedMemoryRegisterRequest, inference.GrpcService.CudaSharedMemoryRegisterResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "CudaSharedMemoryRegister"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  inference.GrpcService.CudaSharedMemoryRegisterRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  inference.GrpcService.CudaSharedMemoryRegisterResponse.getDefaultInstance()))
              .setSchemaDescriptor(new GRPCInferenceServiceMethodDescriptorSupplier("CudaSharedMemoryRegister"))
              .build();
        }
      }
    }
    return getCudaSharedMemoryRegisterMethod;
  }

  private static volatile io.grpc.MethodDescriptor<inference.GrpcService.CudaSharedMemoryUnregisterRequest,
      inference.GrpcService.CudaSharedMemoryUnregisterResponse> getCudaSharedMemoryUnregisterMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "CudaSharedMemoryUnregister",
      requestType = inference.GrpcService.CudaSharedMemoryUnregisterRequest.class,
      responseType = inference.GrpcService.CudaSharedMemoryUnregisterResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<inference.GrpcService.CudaSharedMemoryUnregisterRequest,
      inference.GrpcService.CudaSharedMemoryUnregisterResponse> getCudaSharedMemoryUnregisterMethod() {
    io.grpc.MethodDescriptor<inference.GrpcService.CudaSharedMemoryUnregisterRequest, inference.GrpcService.CudaSharedMemoryUnregisterResponse> getCudaSharedMemoryUnregisterMethod;
    if ((getCudaSharedMemoryUnregisterMethod = GRPCInferenceServiceGrpc.getCudaSharedMemoryUnregisterMethod) == null) {
      synchronized (GRPCInferenceServiceGrpc.class) {
        if ((getCudaSharedMemoryUnregisterMethod = GRPCInferenceServiceGrpc.getCudaSharedMemoryUnregisterMethod) == null) {
          GRPCInferenceServiceGrpc.getCudaSharedMemoryUnregisterMethod = getCudaSharedMemoryUnregisterMethod =
              io.grpc.MethodDescriptor.<inference.GrpcService.CudaSharedMemoryUnregisterRequest, inference.GrpcService.CudaSharedMemoryUnregisterResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "CudaSharedMemoryUnregister"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  inference.GrpcService.CudaSharedMemoryUnregisterRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  inference.GrpcService.CudaSharedMemoryUnregisterResponse.getDefaultInstance()))
              .setSchemaDescriptor(new GRPCInferenceServiceMethodDescriptorSupplier("CudaSharedMemoryUnregister"))
              .build();
        }
      }
    }
    return getCudaSharedMemoryUnregisterMethod;
  }

  private static volatile io.grpc.MethodDescriptor<inference.GrpcService.TraceSettingRequest,
      inference.GrpcService.TraceSettingResponse> getTraceSettingMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "TraceSetting",
      requestType = inference.GrpcService.TraceSettingRequest.class,
      responseType = inference.GrpcService.TraceSettingResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<inference.GrpcService.TraceSettingRequest,
      inference.GrpcService.TraceSettingResponse> getTraceSettingMethod() {
    io.grpc.MethodDescriptor<inference.GrpcService.TraceSettingRequest, inference.GrpcService.TraceSettingResponse> getTraceSettingMethod;
    if ((getTraceSettingMethod = GRPCInferenceServiceGrpc.getTraceSettingMethod) == null) {
      synchronized (GRPCInferenceServiceGrpc.class) {
        if ((getTraceSettingMethod = GRPCInferenceServiceGrpc.getTraceSettingMethod) == null) {
          GRPCInferenceServiceGrpc.getTraceSettingMethod = getTraceSettingMethod =
              io.grpc.MethodDescriptor.<inference.GrpcService.TraceSettingRequest, inference.GrpcService.TraceSettingResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "TraceSetting"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  inference.GrpcService.TraceSettingRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  inference.GrpcService.TraceSettingResponse.getDefaultInstance()))
              .setSchemaDescriptor(new GRPCInferenceServiceMethodDescriptorSupplier("TraceSetting"))
              .build();
        }
      }
    }
    return getTraceSettingMethod;
  }

  private static volatile io.grpc.MethodDescriptor<inference.GrpcService.LogSettingsRequest,
      inference.GrpcService.LogSettingsResponse> getLogSettingsMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "LogSettings",
      requestType = inference.GrpcService.LogSettingsRequest.class,
      responseType = inference.GrpcService.LogSettingsResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<inference.GrpcService.LogSettingsRequest,
      inference.GrpcService.LogSettingsResponse> getLogSettingsMethod() {
    io.grpc.MethodDescriptor<inference.GrpcService.LogSettingsRequest, inference.GrpcService.LogSettingsResponse> getLogSettingsMethod;
    if ((getLogSettingsMethod = GRPCInferenceServiceGrpc.getLogSettingsMethod) == null) {
      synchronized (GRPCInferenceServiceGrpc.class) {
        if ((getLogSettingsMethod = GRPCInferenceServiceGrpc.getLogSettingsMethod) == null) {
          GRPCInferenceServiceGrpc.getLogSettingsMethod = getLogSettingsMethod =
              io.grpc.MethodDescriptor.<inference.GrpcService.LogSettingsRequest, inference.GrpcService.LogSettingsResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "LogSettings"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  inference.GrpcService.LogSettingsRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  inference.GrpcService.LogSettingsResponse.getDefaultInstance()))
              .setSchemaDescriptor(new GRPCInferenceServiceMethodDescriptorSupplier("LogSettings"))
              .build();
        }
      }
    }
    return getLogSettingsMethod;
  }

  /**
   * Creates a new async stub that supports all call types for the service
   */
  public static GRPCInferenceServiceStub newStub(io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<GRPCInferenceServiceStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<GRPCInferenceServiceStub>() {
        @java.lang.Override
        public GRPCInferenceServiceStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new GRPCInferenceServiceStub(channel, callOptions);
        }
      };
    return GRPCInferenceServiceStub.newStub(factory, channel);
  }

  /**
   * Creates a new blocking-style stub that supports unary and streaming output calls on the service
   */
  public static GRPCInferenceServiceBlockingStub newBlockingStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<GRPCInferenceServiceBlockingStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<GRPCInferenceServiceBlockingStub>() {
        @java.lang.Override
        public GRPCInferenceServiceBlockingStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new GRPCInferenceServiceBlockingStub(channel, callOptions);
        }
      };
    return GRPCInferenceServiceBlockingStub.newStub(factory, channel);
  }

  /**
   * Creates a new ListenableFuture-style stub that supports unary calls on the service
   */
  public static GRPCInferenceServiceFutureStub newFutureStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<GRPCInferenceServiceFutureStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<GRPCInferenceServiceFutureStub>() {
        @java.lang.Override
        public GRPCInferenceServiceFutureStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new GRPCInferenceServiceFutureStub(channel, callOptions);
        }
      };
    return GRPCInferenceServiceFutureStub.newStub(factory, channel);
  }

  /**
   * <pre>
   *&#64;&#64;
   *&#64;&#64;.. cpp:var:: service InferenceService
   *&#64;&#64;
   *&#64;&#64;   Inference Server GRPC endpoints.
   *&#64;&#64;
   * </pre>
   */
  public interface AsyncService {

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc ServerLive(ServerLiveRequest) returns
     *&#64;&#64;       (ServerLiveResponse)
     *&#64;&#64;
     *&#64;&#64;     Check liveness of the inference server.
     *&#64;&#64;
     * </pre>
     */
    default void serverLive(inference.GrpcService.ServerLiveRequest request,
        io.grpc.stub.StreamObserver<inference.GrpcService.ServerLiveResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getServerLiveMethod(), responseObserver);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc ServerReady(ServerReadyRequest) returns
     *&#64;&#64;       (ServerReadyResponse)
     *&#64;&#64;
     *&#64;&#64;     Check readiness of the inference server.
     *&#64;&#64;
     * </pre>
     */
    default void serverReady(inference.GrpcService.ServerReadyRequest request,
        io.grpc.stub.StreamObserver<inference.GrpcService.ServerReadyResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getServerReadyMethod(), responseObserver);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc ModelReady(ModelReadyRequest) returns
     *&#64;&#64;       (ModelReadyResponse)
     *&#64;&#64;
     *&#64;&#64;     Check readiness of a model in the inference server.
     *&#64;&#64;
     * </pre>
     */
    default void modelReady(inference.GrpcService.ModelReadyRequest request,
        io.grpc.stub.StreamObserver<inference.GrpcService.ModelReadyResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getModelReadyMethod(), responseObserver);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc ServerMetadata(ServerMetadataRequest) returns
     *&#64;&#64;       (ServerMetadataResponse)
     *&#64;&#64;
     *&#64;&#64;     Get server metadata.
     *&#64;&#64;
     * </pre>
     */
    default void serverMetadata(inference.GrpcService.ServerMetadataRequest request,
        io.grpc.stub.StreamObserver<inference.GrpcService.ServerMetadataResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getServerMetadataMethod(), responseObserver);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc ModelMetadata(ModelMetadataRequest) returns
     *&#64;&#64;       (ModelMetadataResponse)
     *&#64;&#64;
     *&#64;&#64;     Get model metadata.
     *&#64;&#64;
     * </pre>
     */
    default void modelMetadata(inference.GrpcService.ModelMetadataRequest request,
        io.grpc.stub.StreamObserver<inference.GrpcService.ModelMetadataResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getModelMetadataMethod(), responseObserver);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc ModelInfer(ModelInferRequest) returns
     *&#64;&#64;       (ModelInferResponse)
     *&#64;&#64;
     *&#64;&#64;     Perform inference using a specific model.
     *&#64;&#64;
     * </pre>
     */
    default void modelInfer(inference.GrpcService.ModelInferRequest request,
        io.grpc.stub.StreamObserver<inference.GrpcService.ModelInferResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getModelInferMethod(), responseObserver);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc ModelStreamInfer(stream ModelInferRequest) returns
     *&#64;&#64;       (stream ModelStreamInferResponse)
     *&#64;&#64;
     *&#64;&#64;     Perform streaming inference.
     *&#64;&#64;
     * </pre>
     */
    default io.grpc.stub.StreamObserver<inference.GrpcService.ModelInferRequest> modelStreamInfer(
        io.grpc.stub.StreamObserver<inference.GrpcService.ModelStreamInferResponse> responseObserver) {
      return io.grpc.stub.ServerCalls.asyncUnimplementedStreamingCall(getModelStreamInferMethod(), responseObserver);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc ModelConfig(ModelConfigRequest) returns
     *&#64;&#64;       (ModelConfigResponse)
     *&#64;&#64;
     *&#64;&#64;     Get model configuration.
     *&#64;&#64;
     * </pre>
     */
    default void modelConfig(inference.GrpcService.ModelConfigRequest request,
        io.grpc.stub.StreamObserver<inference.GrpcService.ModelConfigResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getModelConfigMethod(), responseObserver);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc ModelStatistics(
     *&#64;&#64;                     ModelStatisticsRequest)
     *&#64;&#64;                   returns (ModelStatisticsResponse)
     *&#64;&#64;
     *&#64;&#64;     Get the cumulative inference statistics for a model.
     *&#64;&#64;
     * </pre>
     */
    default void modelStatistics(inference.GrpcService.ModelStatisticsRequest request,
        io.grpc.stub.StreamObserver<inference.GrpcService.ModelStatisticsResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getModelStatisticsMethod(), responseObserver);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc RepositoryIndex(RepositoryIndexRequest) returns
     *&#64;&#64;       (RepositoryIndexResponse)
     *&#64;&#64;
     *&#64;&#64;     Get the index of model repository contents.
     *&#64;&#64;
     * </pre>
     */
    default void repositoryIndex(inference.GrpcService.RepositoryIndexRequest request,
        io.grpc.stub.StreamObserver<inference.GrpcService.RepositoryIndexResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getRepositoryIndexMethod(), responseObserver);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc RepositoryModelLoad(RepositoryModelLoadRequest) returns
     *&#64;&#64;       (RepositoryModelLoadResponse)
     *&#64;&#64;
     *&#64;&#64;     Load or reload a model from a repository.
     *&#64;&#64;
     * </pre>
     */
    default void repositoryModelLoad(inference.GrpcService.RepositoryModelLoadRequest request,
        io.grpc.stub.StreamObserver<inference.GrpcService.RepositoryModelLoadResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getRepositoryModelLoadMethod(), responseObserver);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc RepositoryModelUnload(RepositoryModelUnloadRequest)
     *&#64;&#64;       returns (RepositoryModelUnloadResponse)
     *&#64;&#64;
     *&#64;&#64;     Unload a model.
     *&#64;&#64;
     * </pre>
     */
    default void repositoryModelUnload(inference.GrpcService.RepositoryModelUnloadRequest request,
        io.grpc.stub.StreamObserver<inference.GrpcService.RepositoryModelUnloadResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getRepositoryModelUnloadMethod(), responseObserver);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc SystemSharedMemoryStatus(
     *&#64;&#64;                     SystemSharedMemoryStatusRequest)
     *&#64;&#64;                   returns (SystemSharedMemoryStatusRespose)
     *&#64;&#64;
     *&#64;&#64;     Get the status of all registered system-shared-memory regions.
     *&#64;&#64;
     * </pre>
     */
    default void systemSharedMemoryStatus(inference.GrpcService.SystemSharedMemoryStatusRequest request,
        io.grpc.stub.StreamObserver<inference.GrpcService.SystemSharedMemoryStatusResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getSystemSharedMemoryStatusMethod(), responseObserver);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc SystemSharedMemoryRegister(
     *&#64;&#64;                     SystemSharedMemoryRegisterRequest)
     *&#64;&#64;                   returns (SystemSharedMemoryRegisterResponse)
     *&#64;&#64;
     *&#64;&#64;     Register a system-shared-memory region.
     *&#64;&#64;
     * </pre>
     */
    default void systemSharedMemoryRegister(inference.GrpcService.SystemSharedMemoryRegisterRequest request,
        io.grpc.stub.StreamObserver<inference.GrpcService.SystemSharedMemoryRegisterResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getSystemSharedMemoryRegisterMethod(), responseObserver);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc SystemSharedMemoryUnregister(
     *&#64;&#64;                     SystemSharedMemoryUnregisterRequest)
     *&#64;&#64;                   returns (SystemSharedMemoryUnregisterResponse)
     *&#64;&#64;
     *&#64;&#64;     Unregister a system-shared-memory region.
     *&#64;&#64;
     * </pre>
     */
    default void systemSharedMemoryUnregister(inference.GrpcService.SystemSharedMemoryUnregisterRequest request,
        io.grpc.stub.StreamObserver<inference.GrpcService.SystemSharedMemoryUnregisterResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getSystemSharedMemoryUnregisterMethod(), responseObserver);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc CudaSharedMemoryStatus(
     *&#64;&#64;                     CudaSharedMemoryStatusRequest)
     *&#64;&#64;                   returns (CudaSharedMemoryStatusRespose)
     *&#64;&#64;
     *&#64;&#64;     Get the status of all registered CUDA-shared-memory regions.
     *&#64;&#64;
     * </pre>
     */
    default void cudaSharedMemoryStatus(inference.GrpcService.CudaSharedMemoryStatusRequest request,
        io.grpc.stub.StreamObserver<inference.GrpcService.CudaSharedMemoryStatusResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getCudaSharedMemoryStatusMethod(), responseObserver);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc CudaSharedMemoryRegister(
     *&#64;&#64;                     CudaSharedMemoryRegisterRequest)
     *&#64;&#64;                   returns (CudaSharedMemoryRegisterResponse)
     *&#64;&#64;
     *&#64;&#64;     Register a CUDA-shared-memory region.
     *&#64;&#64;
     * </pre>
     */
    default void cudaSharedMemoryRegister(inference.GrpcService.CudaSharedMemoryRegisterRequest request,
        io.grpc.stub.StreamObserver<inference.GrpcService.CudaSharedMemoryRegisterResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getCudaSharedMemoryRegisterMethod(), responseObserver);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc CudaSharedMemoryUnregister(
     *&#64;&#64;                     CudaSharedMemoryUnregisterRequest)
     *&#64;&#64;                   returns (CudaSharedMemoryUnregisterResponse)
     *&#64;&#64;
     *&#64;&#64;     Unregister a CUDA-shared-memory region.
     *&#64;&#64;
     * </pre>
     */
    default void cudaSharedMemoryUnregister(inference.GrpcService.CudaSharedMemoryUnregisterRequest request,
        io.grpc.stub.StreamObserver<inference.GrpcService.CudaSharedMemoryUnregisterResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getCudaSharedMemoryUnregisterMethod(), responseObserver);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc TraceSetting(TraceSettingRequest)
     *&#64;&#64;                   returns (TraceSettingResponse)
     *&#64;&#64;
     *&#64;&#64;     Update and get the trace setting of the Triton server.
     *&#64;&#64;
     * </pre>
     */
    default void traceSetting(inference.GrpcService.TraceSettingRequest request,
        io.grpc.stub.StreamObserver<inference.GrpcService.TraceSettingResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getTraceSettingMethod(), responseObserver);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc LogSettings(LogSettingsRequest)
     *&#64;&#64;                   returns (LogSettingsResponse)
     *&#64;&#64;
     *&#64;&#64;     Update and get the log settings of the Triton server.
     *&#64;&#64;
     * </pre>
     */
    default void logSettings(inference.GrpcService.LogSettingsRequest request,
        io.grpc.stub.StreamObserver<inference.GrpcService.LogSettingsResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getLogSettingsMethod(), responseObserver);
    }
  }

  /**
   * Base class for the server implementation of the service GRPCInferenceService.
   * <pre>
   *&#64;&#64;
   *&#64;&#64;.. cpp:var:: service InferenceService
   *&#64;&#64;
   *&#64;&#64;   Inference Server GRPC endpoints.
   *&#64;&#64;
   * </pre>
   */
  public static abstract class GRPCInferenceServiceImplBase
      implements io.grpc.BindableService, AsyncService {

    @java.lang.Override public final io.grpc.ServerServiceDefinition bindService() {
      return GRPCInferenceServiceGrpc.bindService(this);
    }
  }

  /**
   * A stub to allow clients to do asynchronous rpc calls to service GRPCInferenceService.
   * <pre>
   *&#64;&#64;
   *&#64;&#64;.. cpp:var:: service InferenceService
   *&#64;&#64;
   *&#64;&#64;   Inference Server GRPC endpoints.
   *&#64;&#64;
   * </pre>
   */
  public static final class GRPCInferenceServiceStub
      extends io.grpc.stub.AbstractAsyncStub<GRPCInferenceServiceStub> {
    private GRPCInferenceServiceStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected GRPCInferenceServiceStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new GRPCInferenceServiceStub(channel, callOptions);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc ServerLive(ServerLiveRequest) returns
     *&#64;&#64;       (ServerLiveResponse)
     *&#64;&#64;
     *&#64;&#64;     Check liveness of the inference server.
     *&#64;&#64;
     * </pre>
     */
    public void serverLive(inference.GrpcService.ServerLiveRequest request,
        io.grpc.stub.StreamObserver<inference.GrpcService.ServerLiveResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getServerLiveMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc ServerReady(ServerReadyRequest) returns
     *&#64;&#64;       (ServerReadyResponse)
     *&#64;&#64;
     *&#64;&#64;     Check readiness of the inference server.
     *&#64;&#64;
     * </pre>
     */
    public void serverReady(inference.GrpcService.ServerReadyRequest request,
        io.grpc.stub.StreamObserver<inference.GrpcService.ServerReadyResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getServerReadyMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc ModelReady(ModelReadyRequest) returns
     *&#64;&#64;       (ModelReadyResponse)
     *&#64;&#64;
     *&#64;&#64;     Check readiness of a model in the inference server.
     *&#64;&#64;
     * </pre>
     */
    public void modelReady(inference.GrpcService.ModelReadyRequest request,
        io.grpc.stub.StreamObserver<inference.GrpcService.ModelReadyResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getModelReadyMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc ServerMetadata(ServerMetadataRequest) returns
     *&#64;&#64;       (ServerMetadataResponse)
     *&#64;&#64;
     *&#64;&#64;     Get server metadata.
     *&#64;&#64;
     * </pre>
     */
    public void serverMetadata(inference.GrpcService.ServerMetadataRequest request,
        io.grpc.stub.StreamObserver<inference.GrpcService.ServerMetadataResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getServerMetadataMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc ModelMetadata(ModelMetadataRequest) returns
     *&#64;&#64;       (ModelMetadataResponse)
     *&#64;&#64;
     *&#64;&#64;     Get model metadata.
     *&#64;&#64;
     * </pre>
     */
    public void modelMetadata(inference.GrpcService.ModelMetadataRequest request,
        io.grpc.stub.StreamObserver<inference.GrpcService.ModelMetadataResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getModelMetadataMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc ModelInfer(ModelInferRequest) returns
     *&#64;&#64;       (ModelInferResponse)
     *&#64;&#64;
     *&#64;&#64;     Perform inference using a specific model.
     *&#64;&#64;
     * </pre>
     */
    public void modelInfer(inference.GrpcService.ModelInferRequest request,
        io.grpc.stub.StreamObserver<inference.GrpcService.ModelInferResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getModelInferMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc ModelStreamInfer(stream ModelInferRequest) returns
     *&#64;&#64;       (stream ModelStreamInferResponse)
     *&#64;&#64;
     *&#64;&#64;     Perform streaming inference.
     *&#64;&#64;
     * </pre>
     */
    public io.grpc.stub.StreamObserver<inference.GrpcService.ModelInferRequest> modelStreamInfer(
        io.grpc.stub.StreamObserver<inference.GrpcService.ModelStreamInferResponse> responseObserver) {
      return io.grpc.stub.ClientCalls.asyncBidiStreamingCall(
          getChannel().newCall(getModelStreamInferMethod(), getCallOptions()), responseObserver);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc ModelConfig(ModelConfigRequest) returns
     *&#64;&#64;       (ModelConfigResponse)
     *&#64;&#64;
     *&#64;&#64;     Get model configuration.
     *&#64;&#64;
     * </pre>
     */
    public void modelConfig(inference.GrpcService.ModelConfigRequest request,
        io.grpc.stub.StreamObserver<inference.GrpcService.ModelConfigResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getModelConfigMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc ModelStatistics(
     *&#64;&#64;                     ModelStatisticsRequest)
     *&#64;&#64;                   returns (ModelStatisticsResponse)
     *&#64;&#64;
     *&#64;&#64;     Get the cumulative inference statistics for a model.
     *&#64;&#64;
     * </pre>
     */
    public void modelStatistics(inference.GrpcService.ModelStatisticsRequest request,
        io.grpc.stub.StreamObserver<inference.GrpcService.ModelStatisticsResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getModelStatisticsMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc RepositoryIndex(RepositoryIndexRequest) returns
     *&#64;&#64;       (RepositoryIndexResponse)
     *&#64;&#64;
     *&#64;&#64;     Get the index of model repository contents.
     *&#64;&#64;
     * </pre>
     */
    public void repositoryIndex(inference.GrpcService.RepositoryIndexRequest request,
        io.grpc.stub.StreamObserver<inference.GrpcService.RepositoryIndexResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getRepositoryIndexMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc RepositoryModelLoad(RepositoryModelLoadRequest) returns
     *&#64;&#64;       (RepositoryModelLoadResponse)
     *&#64;&#64;
     *&#64;&#64;     Load or reload a model from a repository.
     *&#64;&#64;
     * </pre>
     */
    public void repositoryModelLoad(inference.GrpcService.RepositoryModelLoadRequest request,
        io.grpc.stub.StreamObserver<inference.GrpcService.RepositoryModelLoadResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getRepositoryModelLoadMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc RepositoryModelUnload(RepositoryModelUnloadRequest)
     *&#64;&#64;       returns (RepositoryModelUnloadResponse)
     *&#64;&#64;
     *&#64;&#64;     Unload a model.
     *&#64;&#64;
     * </pre>
     */
    public void repositoryModelUnload(inference.GrpcService.RepositoryModelUnloadRequest request,
        io.grpc.stub.StreamObserver<inference.GrpcService.RepositoryModelUnloadResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getRepositoryModelUnloadMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc SystemSharedMemoryStatus(
     *&#64;&#64;                     SystemSharedMemoryStatusRequest)
     *&#64;&#64;                   returns (SystemSharedMemoryStatusRespose)
     *&#64;&#64;
     *&#64;&#64;     Get the status of all registered system-shared-memory regions.
     *&#64;&#64;
     * </pre>
     */
    public void systemSharedMemoryStatus(inference.GrpcService.SystemSharedMemoryStatusRequest request,
        io.grpc.stub.StreamObserver<inference.GrpcService.SystemSharedMemoryStatusResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getSystemSharedMemoryStatusMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc SystemSharedMemoryRegister(
     *&#64;&#64;                     SystemSharedMemoryRegisterRequest)
     *&#64;&#64;                   returns (SystemSharedMemoryRegisterResponse)
     *&#64;&#64;
     *&#64;&#64;     Register a system-shared-memory region.
     *&#64;&#64;
     * </pre>
     */
    public void systemSharedMemoryRegister(inference.GrpcService.SystemSharedMemoryRegisterRequest request,
        io.grpc.stub.StreamObserver<inference.GrpcService.SystemSharedMemoryRegisterResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getSystemSharedMemoryRegisterMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc SystemSharedMemoryUnregister(
     *&#64;&#64;                     SystemSharedMemoryUnregisterRequest)
     *&#64;&#64;                   returns (SystemSharedMemoryUnregisterResponse)
     *&#64;&#64;
     *&#64;&#64;     Unregister a system-shared-memory region.
     *&#64;&#64;
     * </pre>
     */
    public void systemSharedMemoryUnregister(inference.GrpcService.SystemSharedMemoryUnregisterRequest request,
        io.grpc.stub.StreamObserver<inference.GrpcService.SystemSharedMemoryUnregisterResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getSystemSharedMemoryUnregisterMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc CudaSharedMemoryStatus(
     *&#64;&#64;                     CudaSharedMemoryStatusRequest)
     *&#64;&#64;                   returns (CudaSharedMemoryStatusRespose)
     *&#64;&#64;
     *&#64;&#64;     Get the status of all registered CUDA-shared-memory regions.
     *&#64;&#64;
     * </pre>
     */
    public void cudaSharedMemoryStatus(inference.GrpcService.CudaSharedMemoryStatusRequest request,
        io.grpc.stub.StreamObserver<inference.GrpcService.CudaSharedMemoryStatusResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getCudaSharedMemoryStatusMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc CudaSharedMemoryRegister(
     *&#64;&#64;                     CudaSharedMemoryRegisterRequest)
     *&#64;&#64;                   returns (CudaSharedMemoryRegisterResponse)
     *&#64;&#64;
     *&#64;&#64;     Register a CUDA-shared-memory region.
     *&#64;&#64;
     * </pre>
     */
    public void cudaSharedMemoryRegister(inference.GrpcService.CudaSharedMemoryRegisterRequest request,
        io.grpc.stub.StreamObserver<inference.GrpcService.CudaSharedMemoryRegisterResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getCudaSharedMemoryRegisterMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc CudaSharedMemoryUnregister(
     *&#64;&#64;                     CudaSharedMemoryUnregisterRequest)
     *&#64;&#64;                   returns (CudaSharedMemoryUnregisterResponse)
     *&#64;&#64;
     *&#64;&#64;     Unregister a CUDA-shared-memory region.
     *&#64;&#64;
     * </pre>
     */
    public void cudaSharedMemoryUnregister(inference.GrpcService.CudaSharedMemoryUnregisterRequest request,
        io.grpc.stub.StreamObserver<inference.GrpcService.CudaSharedMemoryUnregisterResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getCudaSharedMemoryUnregisterMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc TraceSetting(TraceSettingRequest)
     *&#64;&#64;                   returns (TraceSettingResponse)
     *&#64;&#64;
     *&#64;&#64;     Update and get the trace setting of the Triton server.
     *&#64;&#64;
     * </pre>
     */
    public void traceSetting(inference.GrpcService.TraceSettingRequest request,
        io.grpc.stub.StreamObserver<inference.GrpcService.TraceSettingResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getTraceSettingMethod(), getCallOptions()), request, responseObserver);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc LogSettings(LogSettingsRequest)
     *&#64;&#64;                   returns (LogSettingsResponse)
     *&#64;&#64;
     *&#64;&#64;     Update and get the log settings of the Triton server.
     *&#64;&#64;
     * </pre>
     */
    public void logSettings(inference.GrpcService.LogSettingsRequest request,
        io.grpc.stub.StreamObserver<inference.GrpcService.LogSettingsResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getLogSettingsMethod(), getCallOptions()), request, responseObserver);
    }
  }

  /**
   * A stub to allow clients to do synchronous rpc calls to service GRPCInferenceService.
   * <pre>
   *&#64;&#64;
   *&#64;&#64;.. cpp:var:: service InferenceService
   *&#64;&#64;
   *&#64;&#64;   Inference Server GRPC endpoints.
   *&#64;&#64;
   * </pre>
   */
  public static final class GRPCInferenceServiceBlockingStub
      extends io.grpc.stub.AbstractBlockingStub<GRPCInferenceServiceBlockingStub> {
    private GRPCInferenceServiceBlockingStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected GRPCInferenceServiceBlockingStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new GRPCInferenceServiceBlockingStub(channel, callOptions);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc ServerLive(ServerLiveRequest) returns
     *&#64;&#64;       (ServerLiveResponse)
     *&#64;&#64;
     *&#64;&#64;     Check liveness of the inference server.
     *&#64;&#64;
     * </pre>
     */
    public inference.GrpcService.ServerLiveResponse serverLive(inference.GrpcService.ServerLiveRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getServerLiveMethod(), getCallOptions(), request);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc ServerReady(ServerReadyRequest) returns
     *&#64;&#64;       (ServerReadyResponse)
     *&#64;&#64;
     *&#64;&#64;     Check readiness of the inference server.
     *&#64;&#64;
     * </pre>
     */
    public inference.GrpcService.ServerReadyResponse serverReady(inference.GrpcService.ServerReadyRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getServerReadyMethod(), getCallOptions(), request);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc ModelReady(ModelReadyRequest) returns
     *&#64;&#64;       (ModelReadyResponse)
     *&#64;&#64;
     *&#64;&#64;     Check readiness of a model in the inference server.
     *&#64;&#64;
     * </pre>
     */
    public inference.GrpcService.ModelReadyResponse modelReady(inference.GrpcService.ModelReadyRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getModelReadyMethod(), getCallOptions(), request);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc ServerMetadata(ServerMetadataRequest) returns
     *&#64;&#64;       (ServerMetadataResponse)
     *&#64;&#64;
     *&#64;&#64;     Get server metadata.
     *&#64;&#64;
     * </pre>
     */
    public inference.GrpcService.ServerMetadataResponse serverMetadata(inference.GrpcService.ServerMetadataRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getServerMetadataMethod(), getCallOptions(), request);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc ModelMetadata(ModelMetadataRequest) returns
     *&#64;&#64;       (ModelMetadataResponse)
     *&#64;&#64;
     *&#64;&#64;     Get model metadata.
     *&#64;&#64;
     * </pre>
     */
    public inference.GrpcService.ModelMetadataResponse modelMetadata(inference.GrpcService.ModelMetadataRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getModelMetadataMethod(), getCallOptions(), request);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc ModelInfer(ModelInferRequest) returns
     *&#64;&#64;       (ModelInferResponse)
     *&#64;&#64;
     *&#64;&#64;     Perform inference using a specific model.
     *&#64;&#64;
     * </pre>
     */
    public inference.GrpcService.ModelInferResponse modelInfer(inference.GrpcService.ModelInferRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getModelInferMethod(), getCallOptions(), request);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc ModelConfig(ModelConfigRequest) returns
     *&#64;&#64;       (ModelConfigResponse)
     *&#64;&#64;
     *&#64;&#64;     Get model configuration.
     *&#64;&#64;
     * </pre>
     */
    public inference.GrpcService.ModelConfigResponse modelConfig(inference.GrpcService.ModelConfigRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getModelConfigMethod(), getCallOptions(), request);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc ModelStatistics(
     *&#64;&#64;                     ModelStatisticsRequest)
     *&#64;&#64;                   returns (ModelStatisticsResponse)
     *&#64;&#64;
     *&#64;&#64;     Get the cumulative inference statistics for a model.
     *&#64;&#64;
     * </pre>
     */
    public inference.GrpcService.ModelStatisticsResponse modelStatistics(inference.GrpcService.ModelStatisticsRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getModelStatisticsMethod(), getCallOptions(), request);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc RepositoryIndex(RepositoryIndexRequest) returns
     *&#64;&#64;       (RepositoryIndexResponse)
     *&#64;&#64;
     *&#64;&#64;     Get the index of model repository contents.
     *&#64;&#64;
     * </pre>
     */
    public inference.GrpcService.RepositoryIndexResponse repositoryIndex(inference.GrpcService.RepositoryIndexRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getRepositoryIndexMethod(), getCallOptions(), request);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc RepositoryModelLoad(RepositoryModelLoadRequest) returns
     *&#64;&#64;       (RepositoryModelLoadResponse)
     *&#64;&#64;
     *&#64;&#64;     Load or reload a model from a repository.
     *&#64;&#64;
     * </pre>
     */
    public inference.GrpcService.RepositoryModelLoadResponse repositoryModelLoad(inference.GrpcService.RepositoryModelLoadRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getRepositoryModelLoadMethod(), getCallOptions(), request);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc RepositoryModelUnload(RepositoryModelUnloadRequest)
     *&#64;&#64;       returns (RepositoryModelUnloadResponse)
     *&#64;&#64;
     *&#64;&#64;     Unload a model.
     *&#64;&#64;
     * </pre>
     */
    public inference.GrpcService.RepositoryModelUnloadResponse repositoryModelUnload(inference.GrpcService.RepositoryModelUnloadRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getRepositoryModelUnloadMethod(), getCallOptions(), request);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc SystemSharedMemoryStatus(
     *&#64;&#64;                     SystemSharedMemoryStatusRequest)
     *&#64;&#64;                   returns (SystemSharedMemoryStatusRespose)
     *&#64;&#64;
     *&#64;&#64;     Get the status of all registered system-shared-memory regions.
     *&#64;&#64;
     * </pre>
     */
    public inference.GrpcService.SystemSharedMemoryStatusResponse systemSharedMemoryStatus(inference.GrpcService.SystemSharedMemoryStatusRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getSystemSharedMemoryStatusMethod(), getCallOptions(), request);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc SystemSharedMemoryRegister(
     *&#64;&#64;                     SystemSharedMemoryRegisterRequest)
     *&#64;&#64;                   returns (SystemSharedMemoryRegisterResponse)
     *&#64;&#64;
     *&#64;&#64;     Register a system-shared-memory region.
     *&#64;&#64;
     * </pre>
     */
    public inference.GrpcService.SystemSharedMemoryRegisterResponse systemSharedMemoryRegister(inference.GrpcService.SystemSharedMemoryRegisterRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getSystemSharedMemoryRegisterMethod(), getCallOptions(), request);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc SystemSharedMemoryUnregister(
     *&#64;&#64;                     SystemSharedMemoryUnregisterRequest)
     *&#64;&#64;                   returns (SystemSharedMemoryUnregisterResponse)
     *&#64;&#64;
     *&#64;&#64;     Unregister a system-shared-memory region.
     *&#64;&#64;
     * </pre>
     */
    public inference.GrpcService.SystemSharedMemoryUnregisterResponse systemSharedMemoryUnregister(inference.GrpcService.SystemSharedMemoryUnregisterRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getSystemSharedMemoryUnregisterMethod(), getCallOptions(), request);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc CudaSharedMemoryStatus(
     *&#64;&#64;                     CudaSharedMemoryStatusRequest)
     *&#64;&#64;                   returns (CudaSharedMemoryStatusRespose)
     *&#64;&#64;
     *&#64;&#64;     Get the status of all registered CUDA-shared-memory regions.
     *&#64;&#64;
     * </pre>
     */
    public inference.GrpcService.CudaSharedMemoryStatusResponse cudaSharedMemoryStatus(inference.GrpcService.CudaSharedMemoryStatusRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getCudaSharedMemoryStatusMethod(), getCallOptions(), request);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc CudaSharedMemoryRegister(
     *&#64;&#64;                     CudaSharedMemoryRegisterRequest)
     *&#64;&#64;                   returns (CudaSharedMemoryRegisterResponse)
     *&#64;&#64;
     *&#64;&#64;     Register a CUDA-shared-memory region.
     *&#64;&#64;
     * </pre>
     */
    public inference.GrpcService.CudaSharedMemoryRegisterResponse cudaSharedMemoryRegister(inference.GrpcService.CudaSharedMemoryRegisterRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getCudaSharedMemoryRegisterMethod(), getCallOptions(), request);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc CudaSharedMemoryUnregister(
     *&#64;&#64;                     CudaSharedMemoryUnregisterRequest)
     *&#64;&#64;                   returns (CudaSharedMemoryUnregisterResponse)
     *&#64;&#64;
     *&#64;&#64;     Unregister a CUDA-shared-memory region.
     *&#64;&#64;
     * </pre>
     */
    public inference.GrpcService.CudaSharedMemoryUnregisterResponse cudaSharedMemoryUnregister(inference.GrpcService.CudaSharedMemoryUnregisterRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getCudaSharedMemoryUnregisterMethod(), getCallOptions(), request);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc TraceSetting(TraceSettingRequest)
     *&#64;&#64;                   returns (TraceSettingResponse)
     *&#64;&#64;
     *&#64;&#64;     Update and get the trace setting of the Triton server.
     *&#64;&#64;
     * </pre>
     */
    public inference.GrpcService.TraceSettingResponse traceSetting(inference.GrpcService.TraceSettingRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getTraceSettingMethod(), getCallOptions(), request);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc LogSettings(LogSettingsRequest)
     *&#64;&#64;                   returns (LogSettingsResponse)
     *&#64;&#64;
     *&#64;&#64;     Update and get the log settings of the Triton server.
     *&#64;&#64;
     * </pre>
     */
    public inference.GrpcService.LogSettingsResponse logSettings(inference.GrpcService.LogSettingsRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getLogSettingsMethod(), getCallOptions(), request);
    }
  }

  /**
   * A stub to allow clients to do ListenableFuture-style rpc calls to service GRPCInferenceService.
   * <pre>
   *&#64;&#64;
   *&#64;&#64;.. cpp:var:: service InferenceService
   *&#64;&#64;
   *&#64;&#64;   Inference Server GRPC endpoints.
   *&#64;&#64;
   * </pre>
   */
  public static final class GRPCInferenceServiceFutureStub
      extends io.grpc.stub.AbstractFutureStub<GRPCInferenceServiceFutureStub> {
    private GRPCInferenceServiceFutureStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected GRPCInferenceServiceFutureStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new GRPCInferenceServiceFutureStub(channel, callOptions);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc ServerLive(ServerLiveRequest) returns
     *&#64;&#64;       (ServerLiveResponse)
     *&#64;&#64;
     *&#64;&#64;     Check liveness of the inference server.
     *&#64;&#64;
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<inference.GrpcService.ServerLiveResponse> serverLive(
        inference.GrpcService.ServerLiveRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getServerLiveMethod(), getCallOptions()), request);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc ServerReady(ServerReadyRequest) returns
     *&#64;&#64;       (ServerReadyResponse)
     *&#64;&#64;
     *&#64;&#64;     Check readiness of the inference server.
     *&#64;&#64;
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<inference.GrpcService.ServerReadyResponse> serverReady(
        inference.GrpcService.ServerReadyRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getServerReadyMethod(), getCallOptions()), request);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc ModelReady(ModelReadyRequest) returns
     *&#64;&#64;       (ModelReadyResponse)
     *&#64;&#64;
     *&#64;&#64;     Check readiness of a model in the inference server.
     *&#64;&#64;
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<inference.GrpcService.ModelReadyResponse> modelReady(
        inference.GrpcService.ModelReadyRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getModelReadyMethod(), getCallOptions()), request);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc ServerMetadata(ServerMetadataRequest) returns
     *&#64;&#64;       (ServerMetadataResponse)
     *&#64;&#64;
     *&#64;&#64;     Get server metadata.
     *&#64;&#64;
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<inference.GrpcService.ServerMetadataResponse> serverMetadata(
        inference.GrpcService.ServerMetadataRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getServerMetadataMethod(), getCallOptions()), request);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc ModelMetadata(ModelMetadataRequest) returns
     *&#64;&#64;       (ModelMetadataResponse)
     *&#64;&#64;
     *&#64;&#64;     Get model metadata.
     *&#64;&#64;
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<inference.GrpcService.ModelMetadataResponse> modelMetadata(
        inference.GrpcService.ModelMetadataRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getModelMetadataMethod(), getCallOptions()), request);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc ModelInfer(ModelInferRequest) returns
     *&#64;&#64;       (ModelInferResponse)
     *&#64;&#64;
     *&#64;&#64;     Perform inference using a specific model.
     *&#64;&#64;
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<inference.GrpcService.ModelInferResponse> modelInfer(
        inference.GrpcService.ModelInferRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getModelInferMethod(), getCallOptions()), request);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc ModelConfig(ModelConfigRequest) returns
     *&#64;&#64;       (ModelConfigResponse)
     *&#64;&#64;
     *&#64;&#64;     Get model configuration.
     *&#64;&#64;
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<inference.GrpcService.ModelConfigResponse> modelConfig(
        inference.GrpcService.ModelConfigRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getModelConfigMethod(), getCallOptions()), request);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc ModelStatistics(
     *&#64;&#64;                     ModelStatisticsRequest)
     *&#64;&#64;                   returns (ModelStatisticsResponse)
     *&#64;&#64;
     *&#64;&#64;     Get the cumulative inference statistics for a model.
     *&#64;&#64;
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<inference.GrpcService.ModelStatisticsResponse> modelStatistics(
        inference.GrpcService.ModelStatisticsRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getModelStatisticsMethod(), getCallOptions()), request);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc RepositoryIndex(RepositoryIndexRequest) returns
     *&#64;&#64;       (RepositoryIndexResponse)
     *&#64;&#64;
     *&#64;&#64;     Get the index of model repository contents.
     *&#64;&#64;
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<inference.GrpcService.RepositoryIndexResponse> repositoryIndex(
        inference.GrpcService.RepositoryIndexRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getRepositoryIndexMethod(), getCallOptions()), request);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc RepositoryModelLoad(RepositoryModelLoadRequest) returns
     *&#64;&#64;       (RepositoryModelLoadResponse)
     *&#64;&#64;
     *&#64;&#64;     Load or reload a model from a repository.
     *&#64;&#64;
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<inference.GrpcService.RepositoryModelLoadResponse> repositoryModelLoad(
        inference.GrpcService.RepositoryModelLoadRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getRepositoryModelLoadMethod(), getCallOptions()), request);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc RepositoryModelUnload(RepositoryModelUnloadRequest)
     *&#64;&#64;       returns (RepositoryModelUnloadResponse)
     *&#64;&#64;
     *&#64;&#64;     Unload a model.
     *&#64;&#64;
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<inference.GrpcService.RepositoryModelUnloadResponse> repositoryModelUnload(
        inference.GrpcService.RepositoryModelUnloadRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getRepositoryModelUnloadMethod(), getCallOptions()), request);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc SystemSharedMemoryStatus(
     *&#64;&#64;                     SystemSharedMemoryStatusRequest)
     *&#64;&#64;                   returns (SystemSharedMemoryStatusRespose)
     *&#64;&#64;
     *&#64;&#64;     Get the status of all registered system-shared-memory regions.
     *&#64;&#64;
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<inference.GrpcService.SystemSharedMemoryStatusResponse> systemSharedMemoryStatus(
        inference.GrpcService.SystemSharedMemoryStatusRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getSystemSharedMemoryStatusMethod(), getCallOptions()), request);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc SystemSharedMemoryRegister(
     *&#64;&#64;                     SystemSharedMemoryRegisterRequest)
     *&#64;&#64;                   returns (SystemSharedMemoryRegisterResponse)
     *&#64;&#64;
     *&#64;&#64;     Register a system-shared-memory region.
     *&#64;&#64;
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<inference.GrpcService.SystemSharedMemoryRegisterResponse> systemSharedMemoryRegister(
        inference.GrpcService.SystemSharedMemoryRegisterRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getSystemSharedMemoryRegisterMethod(), getCallOptions()), request);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc SystemSharedMemoryUnregister(
     *&#64;&#64;                     SystemSharedMemoryUnregisterRequest)
     *&#64;&#64;                   returns (SystemSharedMemoryUnregisterResponse)
     *&#64;&#64;
     *&#64;&#64;     Unregister a system-shared-memory region.
     *&#64;&#64;
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<inference.GrpcService.SystemSharedMemoryUnregisterResponse> systemSharedMemoryUnregister(
        inference.GrpcService.SystemSharedMemoryUnregisterRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getSystemSharedMemoryUnregisterMethod(), getCallOptions()), request);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc CudaSharedMemoryStatus(
     *&#64;&#64;                     CudaSharedMemoryStatusRequest)
     *&#64;&#64;                   returns (CudaSharedMemoryStatusRespose)
     *&#64;&#64;
     *&#64;&#64;     Get the status of all registered CUDA-shared-memory regions.
     *&#64;&#64;
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<inference.GrpcService.CudaSharedMemoryStatusResponse> cudaSharedMemoryStatus(
        inference.GrpcService.CudaSharedMemoryStatusRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getCudaSharedMemoryStatusMethod(), getCallOptions()), request);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc CudaSharedMemoryRegister(
     *&#64;&#64;                     CudaSharedMemoryRegisterRequest)
     *&#64;&#64;                   returns (CudaSharedMemoryRegisterResponse)
     *&#64;&#64;
     *&#64;&#64;     Register a CUDA-shared-memory region.
     *&#64;&#64;
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<inference.GrpcService.CudaSharedMemoryRegisterResponse> cudaSharedMemoryRegister(
        inference.GrpcService.CudaSharedMemoryRegisterRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getCudaSharedMemoryRegisterMethod(), getCallOptions()), request);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc CudaSharedMemoryUnregister(
     *&#64;&#64;                     CudaSharedMemoryUnregisterRequest)
     *&#64;&#64;                   returns (CudaSharedMemoryUnregisterResponse)
     *&#64;&#64;
     *&#64;&#64;     Unregister a CUDA-shared-memory region.
     *&#64;&#64;
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<inference.GrpcService.CudaSharedMemoryUnregisterResponse> cudaSharedMemoryUnregister(
        inference.GrpcService.CudaSharedMemoryUnregisterRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getCudaSharedMemoryUnregisterMethod(), getCallOptions()), request);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc TraceSetting(TraceSettingRequest)
     *&#64;&#64;                   returns (TraceSettingResponse)
     *&#64;&#64;
     *&#64;&#64;     Update and get the trace setting of the Triton server.
     *&#64;&#64;
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<inference.GrpcService.TraceSettingResponse> traceSetting(
        inference.GrpcService.TraceSettingRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getTraceSettingMethod(), getCallOptions()), request);
    }

    /**
     * <pre>
     *&#64;&#64;  .. cpp:var:: rpc LogSettings(LogSettingsRequest)
     *&#64;&#64;                   returns (LogSettingsResponse)
     *&#64;&#64;
     *&#64;&#64;     Update and get the log settings of the Triton server.
     *&#64;&#64;
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<inference.GrpcService.LogSettingsResponse> logSettings(
        inference.GrpcService.LogSettingsRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getLogSettingsMethod(), getCallOptions()), request);
    }
  }

  private static final int METHODID_SERVER_LIVE = 0;
  private static final int METHODID_SERVER_READY = 1;
  private static final int METHODID_MODEL_READY = 2;
  private static final int METHODID_SERVER_METADATA = 3;
  private static final int METHODID_MODEL_METADATA = 4;
  private static final int METHODID_MODEL_INFER = 5;
  private static final int METHODID_MODEL_CONFIG = 6;
  private static final int METHODID_MODEL_STATISTICS = 7;
  private static final int METHODID_REPOSITORY_INDEX = 8;
  private static final int METHODID_REPOSITORY_MODEL_LOAD = 9;
  private static final int METHODID_REPOSITORY_MODEL_UNLOAD = 10;
  private static final int METHODID_SYSTEM_SHARED_MEMORY_STATUS = 11;
  private static final int METHODID_SYSTEM_SHARED_MEMORY_REGISTER = 12;
  private static final int METHODID_SYSTEM_SHARED_MEMORY_UNREGISTER = 13;
  private static final int METHODID_CUDA_SHARED_MEMORY_STATUS = 14;
  private static final int METHODID_CUDA_SHARED_MEMORY_REGISTER = 15;
  private static final int METHODID_CUDA_SHARED_MEMORY_UNREGISTER = 16;
  private static final int METHODID_TRACE_SETTING = 17;
  private static final int METHODID_LOG_SETTINGS = 18;
  private static final int METHODID_MODEL_STREAM_INFER = 19;

  private static final class MethodHandlers<Req, Resp> implements
      io.grpc.stub.ServerCalls.UnaryMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.ServerStreamingMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.ClientStreamingMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.BidiStreamingMethod<Req, Resp> {
    private final AsyncService serviceImpl;
    private final int methodId;

    MethodHandlers(AsyncService serviceImpl, int methodId) {
      this.serviceImpl = serviceImpl;
      this.methodId = methodId;
    }

    @java.lang.Override
    @java.lang.SuppressWarnings("unchecked")
    public void invoke(Req request, io.grpc.stub.StreamObserver<Resp> responseObserver) {
      switch (methodId) {
        case METHODID_SERVER_LIVE:
          serviceImpl.serverLive((inference.GrpcService.ServerLiveRequest) request,
              (io.grpc.stub.StreamObserver<inference.GrpcService.ServerLiveResponse>) responseObserver);
          break;
        case METHODID_SERVER_READY:
          serviceImpl.serverReady((inference.GrpcService.ServerReadyRequest) request,
              (io.grpc.stub.StreamObserver<inference.GrpcService.ServerReadyResponse>) responseObserver);
          break;
        case METHODID_MODEL_READY:
          serviceImpl.modelReady((inference.GrpcService.ModelReadyRequest) request,
              (io.grpc.stub.StreamObserver<inference.GrpcService.ModelReadyResponse>) responseObserver);
          break;
        case METHODID_SERVER_METADATA:
          serviceImpl.serverMetadata((inference.GrpcService.ServerMetadataRequest) request,
              (io.grpc.stub.StreamObserver<inference.GrpcService.ServerMetadataResponse>) responseObserver);
          break;
        case METHODID_MODEL_METADATA:
          serviceImpl.modelMetadata((inference.GrpcService.ModelMetadataRequest) request,
              (io.grpc.stub.StreamObserver<inference.GrpcService.ModelMetadataResponse>) responseObserver);
          break;
        case METHODID_MODEL_INFER:
          serviceImpl.modelInfer((inference.GrpcService.ModelInferRequest) request,
              (io.grpc.stub.StreamObserver<inference.GrpcService.ModelInferResponse>) responseObserver);
          break;
        case METHODID_MODEL_CONFIG:
          serviceImpl.modelConfig((inference.GrpcService.ModelConfigRequest) request,
              (io.grpc.stub.StreamObserver<inference.GrpcService.ModelConfigResponse>) responseObserver);
          break;
        case METHODID_MODEL_STATISTICS:
          serviceImpl.modelStatistics((inference.GrpcService.ModelStatisticsRequest) request,
              (io.grpc.stub.StreamObserver<inference.GrpcService.ModelStatisticsResponse>) responseObserver);
          break;
        case METHODID_REPOSITORY_INDEX:
          serviceImpl.repositoryIndex((inference.GrpcService.RepositoryIndexRequest) request,
              (io.grpc.stub.StreamObserver<inference.GrpcService.RepositoryIndexResponse>) responseObserver);
          break;
        case METHODID_REPOSITORY_MODEL_LOAD:
          serviceImpl.repositoryModelLoad((inference.GrpcService.RepositoryModelLoadRequest) request,
              (io.grpc.stub.StreamObserver<inference.GrpcService.RepositoryModelLoadResponse>) responseObserver);
          break;
        case METHODID_REPOSITORY_MODEL_UNLOAD:
          serviceImpl.repositoryModelUnload((inference.GrpcService.RepositoryModelUnloadRequest) request,
              (io.grpc.stub.StreamObserver<inference.GrpcService.RepositoryModelUnloadResponse>) responseObserver);
          break;
        case METHODID_SYSTEM_SHARED_MEMORY_STATUS:
          serviceImpl.systemSharedMemoryStatus((inference.GrpcService.SystemSharedMemoryStatusRequest) request,
              (io.grpc.stub.StreamObserver<inference.GrpcService.SystemSharedMemoryStatusResponse>) responseObserver);
          break;
        case METHODID_SYSTEM_SHARED_MEMORY_REGISTER:
          serviceImpl.systemSharedMemoryRegister((inference.GrpcService.SystemSharedMemoryRegisterRequest) request,
              (io.grpc.stub.StreamObserver<inference.GrpcService.SystemSharedMemoryRegisterResponse>) responseObserver);
          break;
        case METHODID_SYSTEM_SHARED_MEMORY_UNREGISTER:
          serviceImpl.systemSharedMemoryUnregister((inference.GrpcService.SystemSharedMemoryUnregisterRequest) request,
              (io.grpc.stub.StreamObserver<inference.GrpcService.SystemSharedMemoryUnregisterResponse>) responseObserver);
          break;
        case METHODID_CUDA_SHARED_MEMORY_STATUS:
          serviceImpl.cudaSharedMemoryStatus((inference.GrpcService.CudaSharedMemoryStatusRequest) request,
              (io.grpc.stub.StreamObserver<inference.GrpcService.CudaSharedMemoryStatusResponse>) responseObserver);
          break;
        case METHODID_CUDA_SHARED_MEMORY_REGISTER:
          serviceImpl.cudaSharedMemoryRegister((inference.GrpcService.CudaSharedMemoryRegisterRequest) request,
              (io.grpc.stub.StreamObserver<inference.GrpcService.CudaSharedMemoryRegisterResponse>) responseObserver);
          break;
        case METHODID_CUDA_SHARED_MEMORY_UNREGISTER:
          serviceImpl.cudaSharedMemoryUnregister((inference.GrpcService.CudaSharedMemoryUnregisterRequest) request,
              (io.grpc.stub.StreamObserver<inference.GrpcService.CudaSharedMemoryUnregisterResponse>) responseObserver);
          break;
        case METHODID_TRACE_SETTING:
          serviceImpl.traceSetting((inference.GrpcService.TraceSettingRequest) request,
              (io.grpc.stub.StreamObserver<inference.GrpcService.TraceSettingResponse>) responseObserver);
          break;
        case METHODID_LOG_SETTINGS:
          serviceImpl.logSettings((inference.GrpcService.LogSettingsRequest) request,
              (io.grpc.stub.StreamObserver<inference.GrpcService.LogSettingsResponse>) responseObserver);
          break;
        default:
          throw new AssertionError();
      }
    }

    @java.lang.Override
    @java.lang.SuppressWarnings("unchecked")
    public io.grpc.stub.StreamObserver<Req> invoke(
        io.grpc.stub.StreamObserver<Resp> responseObserver) {
      switch (methodId) {
        case METHODID_MODEL_STREAM_INFER:
          return (io.grpc.stub.StreamObserver<Req>) serviceImpl.modelStreamInfer(
              (io.grpc.stub.StreamObserver<inference.GrpcService.ModelStreamInferResponse>) responseObserver);
        default:
          throw new AssertionError();
      }
    }
  }

  public static final io.grpc.ServerServiceDefinition bindService(AsyncService service) {
    return io.grpc.ServerServiceDefinition.builder(getServiceDescriptor())
        .addMethod(
          getServerLiveMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              inference.GrpcService.ServerLiveRequest,
              inference.GrpcService.ServerLiveResponse>(
                service, METHODID_SERVER_LIVE)))
        .addMethod(
          getServerReadyMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              inference.GrpcService.ServerReadyRequest,
              inference.GrpcService.ServerReadyResponse>(
                service, METHODID_SERVER_READY)))
        .addMethod(
          getModelReadyMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              inference.GrpcService.ModelReadyRequest,
              inference.GrpcService.ModelReadyResponse>(
                service, METHODID_MODEL_READY)))
        .addMethod(
          getServerMetadataMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              inference.GrpcService.ServerMetadataRequest,
              inference.GrpcService.ServerMetadataResponse>(
                service, METHODID_SERVER_METADATA)))
        .addMethod(
          getModelMetadataMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              inference.GrpcService.ModelMetadataRequest,
              inference.GrpcService.ModelMetadataResponse>(
                service, METHODID_MODEL_METADATA)))
        .addMethod(
          getModelInferMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              inference.GrpcService.ModelInferRequest,
              inference.GrpcService.ModelInferResponse>(
                service, METHODID_MODEL_INFER)))
        .addMethod(
          getModelStreamInferMethod(),
          io.grpc.stub.ServerCalls.asyncBidiStreamingCall(
            new MethodHandlers<
              inference.GrpcService.ModelInferRequest,
              inference.GrpcService.ModelStreamInferResponse>(
                service, METHODID_MODEL_STREAM_INFER)))
        .addMethod(
          getModelConfigMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              inference.GrpcService.ModelConfigRequest,
              inference.GrpcService.ModelConfigResponse>(
                service, METHODID_MODEL_CONFIG)))
        .addMethod(
          getModelStatisticsMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              inference.GrpcService.ModelStatisticsRequest,
              inference.GrpcService.ModelStatisticsResponse>(
                service, METHODID_MODEL_STATISTICS)))
        .addMethod(
          getRepositoryIndexMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              inference.GrpcService.RepositoryIndexRequest,
              inference.GrpcService.RepositoryIndexResponse>(
                service, METHODID_REPOSITORY_INDEX)))
        .addMethod(
          getRepositoryModelLoadMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              inference.GrpcService.RepositoryModelLoadRequest,
              inference.GrpcService.RepositoryModelLoadResponse>(
                service, METHODID_REPOSITORY_MODEL_LOAD)))
        .addMethod(
          getRepositoryModelUnloadMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              inference.GrpcService.RepositoryModelUnloadRequest,
              inference.GrpcService.RepositoryModelUnloadResponse>(
                service, METHODID_REPOSITORY_MODEL_UNLOAD)))
        .addMethod(
          getSystemSharedMemoryStatusMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              inference.GrpcService.SystemSharedMemoryStatusRequest,
              inference.GrpcService.SystemSharedMemoryStatusResponse>(
                service, METHODID_SYSTEM_SHARED_MEMORY_STATUS)))
        .addMethod(
          getSystemSharedMemoryRegisterMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              inference.GrpcService.SystemSharedMemoryRegisterRequest,
              inference.GrpcService.SystemSharedMemoryRegisterResponse>(
                service, METHODID_SYSTEM_SHARED_MEMORY_REGISTER)))
        .addMethod(
          getSystemSharedMemoryUnregisterMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              inference.GrpcService.SystemSharedMemoryUnregisterRequest,
              inference.GrpcService.SystemSharedMemoryUnregisterResponse>(
                service, METHODID_SYSTEM_SHARED_MEMORY_UNREGISTER)))
        .addMethod(
          getCudaSharedMemoryStatusMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              inference.GrpcService.CudaSharedMemoryStatusRequest,
              inference.GrpcService.CudaSharedMemoryStatusResponse>(
                service, METHODID_CUDA_SHARED_MEMORY_STATUS)))
        .addMethod(
          getCudaSharedMemoryRegisterMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              inference.GrpcService.CudaSharedMemoryRegisterRequest,
              inference.GrpcService.CudaSharedMemoryRegisterResponse>(
                service, METHODID_CUDA_SHARED_MEMORY_REGISTER)))
        .addMethod(
          getCudaSharedMemoryUnregisterMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              inference.GrpcService.CudaSharedMemoryUnregisterRequest,
              inference.GrpcService.CudaSharedMemoryUnregisterResponse>(
                service, METHODID_CUDA_SHARED_MEMORY_UNREGISTER)))
        .addMethod(
          getTraceSettingMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              inference.GrpcService.TraceSettingRequest,
              inference.GrpcService.TraceSettingResponse>(
                service, METHODID_TRACE_SETTING)))
        .addMethod(
          getLogSettingsMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              inference.GrpcService.LogSettingsRequest,
              inference.GrpcService.LogSettingsResponse>(
                service, METHODID_LOG_SETTINGS)))
        .build();
  }

  private static abstract class GRPCInferenceServiceBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoFileDescriptorSupplier, io.grpc.protobuf.ProtoServiceDescriptorSupplier {
    GRPCInferenceServiceBaseDescriptorSupplier() {}

    @java.lang.Override
    public com.google.protobuf.Descriptors.FileDescriptor getFileDescriptor() {
      return inference.GrpcService.getDescriptor();
    }

    @java.lang.Override
    public com.google.protobuf.Descriptors.ServiceDescriptor getServiceDescriptor() {
      return getFileDescriptor().findServiceByName("GRPCInferenceService");
    }
  }

  private static final class GRPCInferenceServiceFileDescriptorSupplier
      extends GRPCInferenceServiceBaseDescriptorSupplier {
    GRPCInferenceServiceFileDescriptorSupplier() {}
  }

  private static final class GRPCInferenceServiceMethodDescriptorSupplier
      extends GRPCInferenceServiceBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoMethodDescriptorSupplier {
    private final java.lang.String methodName;

    GRPCInferenceServiceMethodDescriptorSupplier(java.lang.String methodName) {
      this.methodName = methodName;
    }

    @java.lang.Override
    public com.google.protobuf.Descriptors.MethodDescriptor getMethodDescriptor() {
      return getServiceDescriptor().findMethodByName(methodName);
    }
  }

  private static volatile io.grpc.ServiceDescriptor serviceDescriptor;

  public static io.grpc.ServiceDescriptor getServiceDescriptor() {
    io.grpc.ServiceDescriptor result = serviceDescriptor;
    if (result == null) {
      synchronized (GRPCInferenceServiceGrpc.class) {
        result = serviceDescriptor;
        if (result == null) {
          serviceDescriptor = result = io.grpc.ServiceDescriptor.newBuilder(SERVICE_NAME)
              .setSchemaDescriptor(new GRPCInferenceServiceFileDescriptorSupplier())
              .addMethod(getServerLiveMethod())
              .addMethod(getServerReadyMethod())
              .addMethod(getModelReadyMethod())
              .addMethod(getServerMetadataMethod())
              .addMethod(getModelMetadataMethod())
              .addMethod(getModelInferMethod())
              .addMethod(getModelStreamInferMethod())
              .addMethod(getModelConfigMethod())
              .addMethod(getModelStatisticsMethod())
              .addMethod(getRepositoryIndexMethod())
              .addMethod(getRepositoryModelLoadMethod())
              .addMethod(getRepositoryModelUnloadMethod())
              .addMethod(getSystemSharedMemoryStatusMethod())
              .addMethod(getSystemSharedMemoryRegisterMethod())
              .addMethod(getSystemSharedMemoryUnregisterMethod())
              .addMethod(getCudaSharedMemoryStatusMethod())
              .addMethod(getCudaSharedMemoryRegisterMethod())
              .addMethod(getCudaSharedMemoryUnregisterMethod())
              .addMethod(getTraceSettingMethod())
              .addMethod(getLogSettingsMethod())
              .build();
        }
      }
    }
    return result;
  }
}
