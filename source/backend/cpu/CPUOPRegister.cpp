// This file is generated by Shell for ops register
namespace MNN {
extern void ___CPUCropAndResizeCreator__OpType_CropAndResize__();
extern void ___CPUConvInt8Creator__OpType_ConvInt8__();
extern void ___CPUArgMaxCreator__OpType_ArgMax__();
extern void ___CPUArgMaxCreator__OpType_ArgMin__();
extern void ___CPUScaleCreator__OpType_Scale__();
extern void ___CPUSelectCreator__OpType_Select__();
extern void ___CPUSoftmaxCreator__OpType_Softmax__();
extern void ___CPUDetectionPostProcessCreator__OpType_DetectionPostProcess__();
extern void ___CPUCastCreator__OpType_Cast__();
extern void ___CPUSoftmaxGradCreator__OpType_SoftmaxGrad__();
extern void ___CPUProposalCreator__OpType_Proposal__();
extern void ___CPUInterpCreator__OpType_Interp__();
extern void ___CPUConstCreator__OpType_Const__();
extern void ___CPUConstCreator__OpType_TrainableParam__();
extern void ___CPUDetectionOutputCreator__OpType_DetectionOutput__();
extern void ___CPUSizeCreator__OpType_Size__();
extern void ___CPUUnravelIndexCreator__OpType_UnravelIndex__();
extern void ___CPUMatMulCreator__OpType_MatMul__();
extern void ___CPUMomentsCreator__OpType_Moments__();
extern void ___CPUInstanceNormCreator__OpType_InstanceNorm__();
extern void ___CPUWhereCreator__OpType_Where__();
extern void ___CPUReluGradCreator__OpType_ReluGrad__();
extern void ___CPUReluGradCreator__OpType_Relu6Grad__();
extern void ___CPUDeconvolutionCreator__OpType_Deconvolution__();
extern void ___CPUBinaryCreator__OpType_BinaryOp__();
extern void ___CPUPoolCreator__OpType_Pooling__();
extern void ___CPUScatterNdCreator__OpType_ScatterNd__();
extern void ___CPUShapeCreator__OpType_Shape__();
extern void ___CPUPluginCreator__OpType_Plugin__();
extern void ___CPUInt8ToFloatCreator__OpType_Int8ToFloat__();
extern void ___CPUROIPoolingCreator__OpType_ROIPooling__();
extern void ___CPUTopKV2Creator__OpType_TopKV2__();
extern void ___CPUUnaryCreator__OpType_UnaryOp__();
extern void ___CPUSigmoidCreator__OpType_Sigmoid__();
extern void ___CPUReductionCreator__OpType_Reduction__();
extern void ___CPUGatherNDCreator__OpType_GatherND__();
extern void ___CPUReluCreator__OpType_ReLU__();
extern void ___CPUReluCreator__OpType_PReLU__();
extern void ___CPURelu6Creator__OpType_ReLU6__();
extern void ___CPUDepthwiseConvInt8Creator__OpType_DepthwiseConvInt8__();
extern void ___CPUOneHotCreator__OpType_OneHot__();
extern void ___CPUPoolInt8Creator__OpType_PoolInt8__();
extern void ___CPUMatrixBandPartCreator__OpType_MatrixBandPart__();
extern void ___CPUDeconvolutionDepthwiseCreator__OpType_DeconvolutionDepthwise__();
extern void ___CPUFloatToInt8Creator__OpType_FloatToInt8__();
extern void ___CPURankCreator__OpType_Rank__();
extern void ___CPULinSpaceCreator__OpType_LinSpace__();
extern void ___CPUNonMaxSuppressionV2Creator__OpType_NonMaxSuppressionV2__();
extern void ___CPUGatherV2Creator__OpType_GatherV2__();
extern void ___CPUGatherV2Creator__OpType_Gather__();
extern void ___CPURasterFactory__OpType_Raster__();
extern void ___CPUConvolutionDepthwiseCreator__OpType_ConvolutionDepthwise__();
extern void ___CPURangeCreator__OpType_Range__();
extern void ___ConvolutionFactory__OpType_Convolution__();
extern void ___CPURNNSequenceGRUCreator__OpType_RNNSequenceGRU__();
extern void ___CPUEltwiseCreator__OpType_Eltwise__();
extern void ___CPUAsStringCreator__OpType_AsString__();
extern void ___CPURandomUniformCreator__OpType_RandomUniform__();
extern void ___CPUSetDiff1DCreator__OpType_SetDiff1D__();
extern void ___CPUReduceJoinCreator__OpType_ReduceJoin__();
extern void ___CPUPriorBoxCreator__OpType_PriorBox__();
extern void ___CPUEltwiseInt8Creator__OpType_EltwiseInt8__();
extern void ___CPUBatchMatMulCreator__OpType_BatchMatMul__();
extern void ___CPULayerNormCreator__OpType_LayerNorm__();
#ifdef MNN_SUPPORT_TFLITE_QUAN
extern void ___CPUQuantizedLogisticCreator__OpType_QuantizedLogistic__();
extern void ___CPUQuantizedMaxPoolCreator__OpType_QuantizedMaxPool__();
extern void ___CPUDepthwiseCreator__OpType_QuantizedDepthwiseConv2D__();
extern void ___CPUQuantizedSoftmaxCreator__OpType_QuantizedSoftmax__();
extern void ___CPUQuantizedAddCreator__OpType_QuantizedAdd__();
extern void ___CPUDequantizeCreator__OpType_Dequantize__();
extern void ___CPUTFQuantizedConv2DCreator__OpType_TfQuantizedConv2D__();
extern void ___CPUQuantizedAvgPoolCreator__OpType_QuantizedAvgPool__();
#endif

void registerCPUOps() {
___CPUCropAndResizeCreator__OpType_CropAndResize__();
___CPUConvInt8Creator__OpType_ConvInt8__();
___CPUArgMaxCreator__OpType_ArgMax__();
___CPUArgMaxCreator__OpType_ArgMin__();
___CPUScaleCreator__OpType_Scale__();
___CPUSelectCreator__OpType_Select__();
___CPUSoftmaxCreator__OpType_Softmax__();
___CPUDetectionPostProcessCreator__OpType_DetectionPostProcess__();
___CPUCastCreator__OpType_Cast__();
___CPUSoftmaxGradCreator__OpType_SoftmaxGrad__();
___CPUProposalCreator__OpType_Proposal__();
___CPUInterpCreator__OpType_Interp__();
___CPUConstCreator__OpType_Const__();
___CPUConstCreator__OpType_TrainableParam__();
___CPUDetectionOutputCreator__OpType_DetectionOutput__();
___CPUSizeCreator__OpType_Size__();
___CPUUnravelIndexCreator__OpType_UnravelIndex__();
___CPUMatMulCreator__OpType_MatMul__();
___CPUMomentsCreator__OpType_Moments__();
___CPUInstanceNormCreator__OpType_InstanceNorm__();
___CPUWhereCreator__OpType_Where__();
___CPUReluGradCreator__OpType_ReluGrad__();
___CPUReluGradCreator__OpType_Relu6Grad__();
___CPUDeconvolutionCreator__OpType_Deconvolution__();
___CPUBinaryCreator__OpType_BinaryOp__();
___CPUPoolCreator__OpType_Pooling__();
___CPUScatterNdCreator__OpType_ScatterNd__();
___CPUShapeCreator__OpType_Shape__();
___CPUPluginCreator__OpType_Plugin__();
___CPUInt8ToFloatCreator__OpType_Int8ToFloat__();
___CPUROIPoolingCreator__OpType_ROIPooling__();
___CPUTopKV2Creator__OpType_TopKV2__();
___CPUUnaryCreator__OpType_UnaryOp__();
___CPUSigmoidCreator__OpType_Sigmoid__();
___CPUReductionCreator__OpType_Reduction__();
___CPUGatherNDCreator__OpType_GatherND__();
___CPUReluCreator__OpType_ReLU__();
___CPUReluCreator__OpType_PReLU__();
___CPURelu6Creator__OpType_ReLU6__();
___CPUDepthwiseConvInt8Creator__OpType_DepthwiseConvInt8__();
___CPUOneHotCreator__OpType_OneHot__();
___CPUPoolInt8Creator__OpType_PoolInt8__();
___CPUMatrixBandPartCreator__OpType_MatrixBandPart__();
___CPUDeconvolutionDepthwiseCreator__OpType_DeconvolutionDepthwise__();
___CPUFloatToInt8Creator__OpType_FloatToInt8__();
___CPURankCreator__OpType_Rank__();
___CPULinSpaceCreator__OpType_LinSpace__();
___CPUNonMaxSuppressionV2Creator__OpType_NonMaxSuppressionV2__();
___CPUGatherV2Creator__OpType_GatherV2__();
___CPUGatherV2Creator__OpType_Gather__();
___CPURasterFactory__OpType_Raster__();
___CPUConvolutionDepthwiseCreator__OpType_ConvolutionDepthwise__();
___CPURangeCreator__OpType_Range__();
___ConvolutionFactory__OpType_Convolution__();
___CPURNNSequenceGRUCreator__OpType_RNNSequenceGRU__();
___CPUEltwiseCreator__OpType_Eltwise__();
___CPUAsStringCreator__OpType_AsString__();
___CPURandomUniformCreator__OpType_RandomUniform__();
___CPUSetDiff1DCreator__OpType_SetDiff1D__();
___CPUReduceJoinCreator__OpType_ReduceJoin__();
___CPUPriorBoxCreator__OpType_PriorBox__();
___CPUEltwiseInt8Creator__OpType_EltwiseInt8__();
___CPUBatchMatMulCreator__OpType_BatchMatMul__();
___CPULayerNormCreator__OpType_LayerNorm__();
#ifdef MNN_SUPPORT_TFLITE_QUAN
___CPUQuantizedLogisticCreator__OpType_QuantizedLogistic__();
___CPUQuantizedMaxPoolCreator__OpType_QuantizedMaxPool__();
___CPUDepthwiseCreator__OpType_QuantizedDepthwiseConv2D__();
___CPUQuantizedSoftmaxCreator__OpType_QuantizedSoftmax__();
___CPUQuantizedAddCreator__OpType_QuantizedAdd__();
___CPUDequantizeCreator__OpType_Dequantize__();
___CPUTFQuantizedConv2DCreator__OpType_TfQuantizedConv2D__();
___CPUQuantizedAvgPoolCreator__OpType_QuantizedAvgPool__();
#endif
}
}
