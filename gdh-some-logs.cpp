
#ifdef MNN_DEBUG_TENSOR_SIZE
    MNN_PRINT("\t===> compute shape: %s, [%s]\n", mOriginOp->name()->c_str(), MNN::EnumNameOpType(mOriginOp->type()));
    if (mInputs.size()) {
        MNN_PRINT("Inputs:\n");
        for (auto o : mInputs) {
            if (o->dimensions() == 0) {
                MNN_PRINT("\t*Scalar*");
            }
            for (int i = 0; i < o->dimensions(); ++i) {
                MNN_PRINT("%d, ", o->length(i));
            }
            MNN_PRINT("\n");
        }
    }
    MNN_PRINT("Outputs:\n");
    for (auto o : mOutputs) {
        if (o->dimensions() == 0) {
            MNN_PRINT("\t*Scalar* ");
        }
        for (int i = 0; i < o->dimensions(); ++i) {
            MNN_PRINT("%d, ", o->length(i));
        }
        MNN_PRINT("format=%d\n", TensorUtils::getDescribe(o)->dimensionFormat);
        MNN_PRINT("\n");
    }
#endif



    for (auto& u : mUnits) {
        auto name1 = u->mOriginOp->name()->str();
        printf("[prep] unit #%s\n", name1.c_str());
        std::vector<std::string> inputs, outputs;
        int i1 = 0;
        for (auto input : u->mInputs) {
            auto ptr = input->buffer().dim;
            std::ostringstream format;
            format << "input_idx=" << u->mOriginOp->inputIndexes()->data()[i1++] << ": ";
            for (int i = 0; i < input->buffer().dimensions; i++) {
            format << ptr[i].extent << ", ";
            }
            format << "unit_type=" << (int) input->getType().code;
            inputs.push_back(format.str());
            printf("\t%s\n", inputs.end()[-1].c_str());
        }
        i1 = 0;
        for (auto output : u->mOutputs) {
            auto ptr = output->buffer().dim;
            std::ostringstream format;
            format << "output_idx=" << u->mOriginOp->outputIndexes()->data()[i1++] << ": ";
            for (int i = 0; i < output->buffer().dimensions; i++) {
            format << ptr[i].extent << ", ";
            }
            format << "unit_type=" << (int)output->getType().code;
            outputs.push_back(format.str());
            printf("\t%s\n", outputs.end()[-1].c_str());
        }
        auto code = u->prepare(mBackend, mBackupBackend);
        if (NO_ERROR != code) {
            if (nullptr != u->mOriginOp->name()) {
                MNN_PRINT("-----------------------------------------------------------------------------------------------------------------------------\n");
                MNN_PRINT("due to the internal logic of MNN, if your MNN model doesn't have input shape, you may ignore this 'Resize error' information:\n");
                MNN_ERROR("** Resize error for [%s], %s, code=%d **\n", MNN::EnumNameOpType(u->mOriginOp->type()),u->mOriginOp->name()->c_str(), code);
                MNN_PRINT("it will work after you set the input tensor shape in MNN, and then resize the Session\n");
                MNN_PRINT("-----------------------------------------------------------------------------------------------------------------------------\n");
            }
            return code;