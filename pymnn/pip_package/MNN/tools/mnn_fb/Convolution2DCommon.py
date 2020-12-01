# automatically generated by the FlatBuffers compiler, do not modify

# namespace: MNN

import flatbuffers

class Convolution2DCommon(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsConvolution2DCommon(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Convolution2DCommon()
        x.Init(buf, n + offset)
        return x

    # Convolution2DCommon
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Convolution2DCommon
    def PadX(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # Convolution2DCommon
    def PadY(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # Convolution2DCommon
    def KernelX(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 1

    # Convolution2DCommon
    def KernelY(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 1

    # Convolution2DCommon
    def StrideX(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 1

    # Convolution2DCommon
    def StrideY(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 1

    # Convolution2DCommon
    def DilateX(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 1

    # Convolution2DCommon
    def DilateY(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 1

    # Convolution2DCommon
    def PadMode(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # Convolution2DCommon
    def Group(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(22))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 1

    # Convolution2DCommon
    def OutputCount(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(24))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # Convolution2DCommon
    def InputCount(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(26))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # Convolution2DCommon
    def Relu(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(28))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

    # Convolution2DCommon
    def Relu6(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(30))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

    # Convolution2DCommon
    def Pads(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(32))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # Convolution2DCommon
    def PadsAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(32))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
        return 0

    # Convolution2DCommon
    def PadsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(32))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

def Convolution2DCommonStart(builder): builder.StartObject(15)
def Convolution2DCommonAddPadX(builder, padX): builder.PrependInt32Slot(0, padX, 0)
def Convolution2DCommonAddPadY(builder, padY): builder.PrependInt32Slot(1, padY, 0)
def Convolution2DCommonAddKernelX(builder, kernelX): builder.PrependInt32Slot(2, kernelX, 1)
def Convolution2DCommonAddKernelY(builder, kernelY): builder.PrependInt32Slot(3, kernelY, 1)
def Convolution2DCommonAddStrideX(builder, strideX): builder.PrependInt32Slot(4, strideX, 1)
def Convolution2DCommonAddStrideY(builder, strideY): builder.PrependInt32Slot(5, strideY, 1)
def Convolution2DCommonAddDilateX(builder, dilateX): builder.PrependInt32Slot(6, dilateX, 1)
def Convolution2DCommonAddDilateY(builder, dilateY): builder.PrependInt32Slot(7, dilateY, 1)
def Convolution2DCommonAddPadMode(builder, padMode): builder.PrependInt8Slot(8, padMode, 0)
def Convolution2DCommonAddGroup(builder, group): builder.PrependInt32Slot(9, group, 1)
def Convolution2DCommonAddOutputCount(builder, outputCount): builder.PrependInt32Slot(10, outputCount, 0)
def Convolution2DCommonAddInputCount(builder, inputCount): builder.PrependInt32Slot(11, inputCount, 0)
def Convolution2DCommonAddRelu(builder, relu): builder.PrependBoolSlot(12, relu, 0)
def Convolution2DCommonAddRelu6(builder, relu6): builder.PrependBoolSlot(13, relu6, 0)
def Convolution2DCommonAddPads(builder, pads): builder.PrependUOffsetTRelativeSlot(14, flatbuffers.number_types.UOffsetTFlags.py_type(pads), 0)
def Convolution2DCommonStartPadsVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def Convolution2DCommonEnd(builder): return builder.EndObject()
