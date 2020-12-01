//
//  Schedule.cpp
//  MNN
//
//  Created by MNN on 2018/07/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "core/Schedule.hpp"
#include <algorithm>
#include <iterator>
#include <set>
#include <unordered_map>
#include "core/DirectedAcyclicGraph.hpp"
#include "core/Macro.h"
#include "core/RuntimeFactory.hpp"
#include "core/TensorUtils.hpp"
#include "shape/SizeComputer.hpp"
#include "utils/InitNet.hpp"
//#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
//#define MNN_AUTO_CHECK_COST
namespace MNN {

class OpNodeDef : public NodeDef<Op*> {
public:
    OpNodeDef(Op* op) {
        this->op = op;
    }

public:
    virtual shared_ptr<Node<Op*>> makeNode() override {
        shared_ptr<Node<Op*>> ptr = make_shared<Node<Op*>>();
        ptr->setData(this->op);
        return ptr;
    }

private:
    Op* op;
};

MNNForwardType Schedule::getApprociateType(const ScheduleConfig& config) {
    MNNForwardType type = config.type;
    // FIXME: Support Auto determine
    if (MNN_FORWARD_AUTO == config.type) {
        // Search Backend Exclude MNN_FORWARD_CPU
        for (int i = 1; i < MNN_FORWARD_ALL; ++i) {
            if (MNNGetExtraRuntimeCreator((MNNForwardType)i) != nullptr) {
                type = (MNNForwardType)i;
                break;
            }
        }
    }
    auto creator = MNNGetExtraRuntimeCreator(type);
    if (nullptr == creator) {
        MNN_PRINT("Can't Find type=%d backend, use %d instead\n", type, config.backupType);
        type = config.backupType;
    }
    return type;
}

static bool _setUpTensorInfo(std::vector<std::shared_ptr<Tensor>>& allTensors, const Net* net) {
    bool valid    = true;
    auto& tensors = allTensors;
    tensors.resize(net->tensorName()->size());
    if (net->usage() == Usage_INFERENCE_STATIC) {
        // static model will set all tensors' shape
        auto describes = net->extraTensorDescribe();
        std::vector<const TensorDescribe*> des(tensors.size());
        for (int i = 0; i < describes->size(); i++) {
            int index  = describes->GetAs<TensorDescribe>(i)->index();
            des[index] = describes->GetAs<TensorDescribe>(i);
        }
        for (int i = 0; i < tensors.size(); ++i) {
            auto blob = des[i]->blob();
            if (auto idims = blob->dims()) {
                tensors[i].reset(new Tensor(idims->size()));
                auto& tb = tensors[i]->buffer();
                for (int d = 0; d < idims->size(); d++) {
                    tb.dim[d].extent = idims->Get(d);
                }
            } else {
                tensors[i].reset(new Tensor(1));
            }
            tensors[i]->setType(blob->dataType());
        }
        for (int i = 0; i < tensors.size(); ++i) {
            auto blob                                                   = des[i]->blob();
            TensorUtils::getDescribe(tensors[i].get())->dimensionFormat = blob->dataFormat();
            if (auto regions = des[i]->regions()) {
                auto& regs = TensorUtils::getDescribe(tensors[i].get())->regions;
                TensorUtils::getDescribe(tensors[i].get())->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
                regs.reserve(regions->size());
                for (int r = 0; r < regions->size(); r++) {
                    auto region = regions->GetAs<Region>(r);
                    Tensor::InsideDescribe::Region reg;
                    reg.origin     = tensors[region->origin()].get();
                    reg.src.offset = region->src()->offset();
                    reg.dst.offset = region->dst()->offset();
                    for (int d = 0; d < 3; d++) {
                        reg.size[d]       = region->size()->data()[d];
                        reg.src.stride[d] = region->src()->stride()->data()[d];
                        reg.dst.stride[d] = region->dst()->stride()->data()[d];
                    }
                    regs.emplace_back(std::move(reg));
                }
            }
        }
        for (int opIndex = 0; opIndex < net->oplists()->size(); ++opIndex) {
            auto op = net->oplists()->GetAs<Op>(opIndex);
            if (OpType_Const == op->type()) {
                MNN_ASSERT(nullptr != op->outputIndexes());
                auto index                                            = op->outputIndexes()->data()[0];
                TensorUtils::getDescribe(tensors[index].get())->usage = Tensor::InsideDescribe::CONSTANT;
            }
        }
    } else {
        // Dynamic Model just set input tensor's shape
        valid = initTensors(tensors, net);
    }
    return valid;
}

static int _findOpPosition(const std::string& opName, const Net* net) {
    for (int i = 0; i < net->oplists()->size(); ++i) {
        auto op = net->oplists()->GetAs<Op>(i);
        if (opName == op->name()->str()) {
            return i;
        }
    }
    return -1;
}

static bool _validateOp(const Op* op) {
    if (nullptr == op->inputIndexes() && nullptr == op->outputIndexes()) {
        return false;
    }
    if (nullptr == op->name()) {
        return false;
    }
    return true;
}

static vector<Op*> generateOneSchedulePath(const Net* net, const vector<int> &starts, const int top,
                                           const vector<shared_ptr<Tensor>>& allTensors,
                                           const std::unordered_map<int, const Op*>& tensorIndexOpMap,
                                           const std::unordered_map<const Op*, int>& opIndexMap) {
    vector<Op*> oplists;
    if (top < 0) {
      int end = (int)net->oplists()->size();
      for (int i = starts.empty() ? 0 : *std::min_element(starts.begin(), starts.end()); i < end; ++i) {
        auto op = net->oplists()->GetAs<Op>(i);
        if (op->type() == OpType_Input || !_validateOp(op)) {
          continue;
        }
        oplists.emplace_back(const_cast<Op*>(op));
      }
    }
    else {
      std::set<const Op*> knownOps;
      auto topOp = net->oplists()->GetAs<Op>(top);
      if (topOp->type() != OpType_Input && _validateOp(topOp)) {
        oplists.emplace_back(const_cast<Op*>(topOp));
        knownOps.insert(topOp);
      }
      for (auto input: starts) {
        knownOps.insert(net->oplists()->GetAs<Op>(input));
      }
      for (size_t i = 0; i < oplists.size(); i++) {
        auto *output = oplists[i];
        if (output->type() == OpType_Input || !_validateOp(output)) {
          continue;
        }
        if (nullptr != output->inputIndexes()) {
          auto data = output->inputIndexes()->data();
          for (int j = 0, end = (int)output->inputIndexes()->size(); j < end; ++j) {
            auto *op = tensorIndexOpMap.at(data[j]);
            if (op->type() == OpType_Input || !_validateOp(op)) {
              continue;
            }
            if (knownOps.find(op) == knownOps.end()) {
              oplists.insert(oplists.begin() + i + 1, const_cast<Op*>(op));
              knownOps.insert(op);
            }
          }
        }
      }
      std::reverse(oplists.begin(), oplists.end());
      std::sort(oplists.begin(), oplists.end(), [&opIndexMap](auto a, auto b) -> bool {
        return opIndexMap.at(a) < opIndexMap.at(b);
      });
    }
    return oplists;
}

static vector<vector<Op*>> generateSchedulePath(const Net* net, const ScheduleConfig& config,
                                                const vector<shared_ptr<Tensor>>& allTensors,
                                                const std::unordered_map<int, const Op*>& tensorIndexOpMap,
                                                const std::unordered_map<const Op*, int>& opIndexMap) {
    vector<vector<Op*>> oplists;

    vector<int> starts;
    for (auto &in: config.path.inputs) {
        if (in.length() > 0) {
            auto pos = _findOpPosition(in, net);
            if (-1 == pos) {
                MNN_PRINT("Can't find %s op as start op\n", in.c_str());
                return oplists;
            }
            else {
              starts.push_back(pos);
            }
        }
    }
    for (auto &out: config.path.outputs) {
        int top = -1;
        if (out.length() > 0) {
            auto pos = _findOpPosition(out, net);
            if (-1 == pos) {
                MNN_PRINT("Can't find %s op as end op\n", out.c_str());
                oplists.clear();
                break;
            } else {
                top = pos;
            }
        }
        vector<Op*> path = generateOneSchedulePath(net, starts, top, allTensors, tensorIndexOpMap, opIndexMap);
        oplists.emplace_back(path);
    }

    return oplists;
}

static void generateScheduleGraph(vector<const Op*>& ops, const Net* net, const ScheduleConfig& configs,
                                  const vector<shared_ptr<Tensor>>& allTensors) {
    if (configs.path.inputs.empty() && configs.path.outputs.empty()) {
        // Use Default Linear schedule
        ops.clear();
        ops.reserve(net->oplists()->size());
        for (int i = 0; i < net->oplists()->size(); ++i) {
            auto op = net->oplists()->GetAs<Op>(i);
            if (op->type() != OpType_Input) {
                ops.emplace_back(op);
            }
        }
        return;
    }
    std::unordered_map<int, const Op*> tensorIndexOpMap;
    for (int i = 0; i < net->oplists()->size(); i++) {
      auto *op = net->oplists()->GetAs<Op>(i);
      if (op->outputIndexes() != nullptr) {
        auto *data = op->outputIndexes()->data();
        for (auto j = 0; j < op->outputIndexes()->size(); j++) {
          tensorIndexOpMap[data[j]] = op;
        }
      }
    }
    std::unordered_map<const Op*, int> opIndexMap;
    for (int i = 0; i < net->oplists()->size(); i++) {
      opIndexMap[net->oplists()->GetAs<Op>(i)] = i;
    }
    vector<vector<Op*>> paths = generateSchedulePath(net, configs, allTensors, tensorIndexOpMap, opIndexMap);

    unique_ptr<DirectedAcyclicGraph<Op*>> graph(new DirectedAcyclicGraph<Op*>());

    // add Node
    unordered_map<const Op*, shared_ptr<Node<Op*>>> opMaps;
    for (vector<Op*> path : paths) {
        for (Op* op : path) {
            if (opMaps.find(op) == opMaps.end()) {
                OpNodeDef def(op);
                shared_ptr<Node<Op*>> n = graph->AddNode(def);
                opMaps.insert(make_pair(op, n));
            }
        }
    }

    // add edges
    for (auto iter: opMaps) {
        auto *op = iter.first;
        if (op->inputIndexes() != nullptr) {
            auto data = op->inputIndexes()->data();
            for (int j = 0, end = (int)op->inputIndexes()->size(); j < end; ++j) {
                auto *inputOp = tensorIndexOpMap[data[j]];
                auto preIter = opMaps.find(inputOp);
                if (preIter != opMaps.end()) {
                  graph->AddEdge(preIter->second, iter.second);
                }
            }
        }
    }
    ops.clear();
    vector<shared_ptr<Node<Op*>>> order;
    if (graph->GetPostOrder(order)) {
        for (shared_ptr<Node<Op*>> n : order) {
            ops.emplace_back(n->getData());
        }
    } else {
        MNN_PRINT("op graph have cycle,schedule failed\n");
    }
}

static vector<Schedule::PipelineInfo> _scheduleUnit(const Net* net, const ScheduleConfig& configs,
                                                    const vector<shared_ptr<Tensor>>& allTensors) {
    vector<Schedule::PipelineInfo> oplists;
    vector<const Op*> ops;
    generateScheduleGraph(ops, net, configs, allTensors);
    initPipelineInfosFromOps(oplists, ops, allTensors);
    return oplists;
}

Schedule::ScheduleInfo Schedule::schedule(const Net* net, const std::vector<ScheduleConfig>& configs) {
    std::vector<std::shared_ptr<Tensor>> allTensors;

    ScheduleInfo schedule;
    if (nullptr == net->oplists()) {
        MNN_PRINT("Error net for schedule\n");
        return schedule;
    }
    bool valid              = _setUpTensorInfo(allTensors, net);
    schedule.validForResize = valid;

    std::vector<std::pair<Backend::Info, std::vector<Schedule::PipelineInfo>>> result;

    for (auto& config : configs) {
        Backend::Info compute;
        compute.type      = getApprociateType(config);
        compute.numThread = config.numThread;
        compute.user      = config.backendConfig;
        auto oplists      = _scheduleUnit(net, config, allTensors);
        result.emplace_back(std::make_pair(compute, std::move(oplists)));
    }

    schedule.pipelineInfo = std::move(result);

    // get all used op's output, drop unused op, won't change op order. always insert all Input Ops
    std::vector<const Op*> oplists;
    {
        for (std::pair<Backend::Info, vector<Schedule::PipelineInfo>>& pipeline : schedule.pipelineInfo) {
            for (auto& info : pipeline.second) {
                oplists.push_back(info.op);
            }
        }
    }
    // set tensors' input/output usage by oplists info
    setInputOutputForOps(allTensors, oplists, net->usage() == Usage_INFERENCE_STATIC);

    // add output index by config info and outputName
    std::unordered_map<std::string, int> tensorNameIndexMap;
    for (int i = 0; i < net->tensorName()->size(); ++i) {
        tensorNameIndexMap[net->tensorName()->Get(i)->str()] = i;
    }
    std::unordered_map<std::string, const Op *> opNameIndexMap;
    for (int i = 0; i < net->oplists()->size(); ++i) {
        auto op = net->oplists()->GetAs<Op>(i);
        opNameIndexMap[op->name()->str()] = op;
    }
    bool hasCustomOutputs = false;
    for (auto& config : configs) {
        for (const auto& name : config.saveTensors) {
            auto iter = tensorNameIndexMap.find(name);
            if (iter != tensorNameIndexMap.end()) {
                auto t = allTensors[iter->second].get();
                if (TensorUtils::getDescribe(t)->usage == Tensor::InsideDescribe::NORMAL) {
                    TensorUtils::getDescribe(t)->usage = Tensor::InsideDescribe::OUTPUT;
                } else {
                    schedule.outputTensor.insert(
                               std::make_pair(net->tensorName()->GetAsString(iter->second)->c_str(), t));
                }
            } else {
                MNN_PRINT("Bad outputname: %s\n", name.c_str());
            }
        }
        for (auto& name : config.path.outputs) {
            auto op = opNameIndexMap.find(name)->second;
            if (op != nullptr && op->outputIndexes() != nullptr) {
                auto *data = op->outputIndexes()->data();
                for (int i = 0; i < op->outputIndexes()->size(); i++) {
                    auto outputIndex = data[i];
                    auto t = allTensors[outputIndex].get();
                    if (TensorUtils::getDescribe(t)->usage == Tensor::InsideDescribe::NORMAL) {
                      TensorUtils::getDescribe(t)->usage = Tensor::InsideDescribe::OUTPUT;
                    }
                    else {
                      schedule.outputTensor.insert(
                        std::make_pair(net->tensorName()->GetAsString(outputIndex)->c_str(), t));
                    }
                    hasCustomOutputs = true;
                }
            }
        }
    }
    if (!hasCustomOutputs && net->outputName() != nullptr) {
        for (int i = 0; i < net->outputName()->size(); ++i) {
            std::string name = net->outputName()->Get(i)->str();
            auto iter = tensorNameIndexMap.find(name);
            if (iter != tensorNameIndexMap.end()) {
                auto t = allTensors[iter->second].get();
                if (TensorUtils::getDescribe(t)->usage == Tensor::InsideDescribe::NORMAL) {
                    TensorUtils::getDescribe(t)->usage = Tensor::InsideDescribe::OUTPUT;
                } else {
                    schedule.outputTensor.insert(
                               std::make_pair(net->tensorName()->GetAsString(iter->second)->c_str(), t));
                }
            }
        }
    }
    // add input/output tensor to schedule's input/output
    for (int index = 0; index < allTensors.size(); index++) {
        auto t = allTensors[index].get();
        auto usage = TensorUtils::getDescribe(t)->usage;
        if (usage == Tensor::InsideDescribe::INPUT) {
            schedule.inputTensors.insert(std::make_pair(net->tensorName()->GetAsString(index)->c_str(), t));
        }
        if (usage == Tensor::InsideDescribe::OUTPUT) {
            schedule.outputTensor.insert(
                       std::make_pair(net->tensorName()->GetAsString(index)->c_str(), t));
        }
    }
    // move tensors to schedule
    for (auto& t : allTensors) {
        schedule.allTensors.emplace_back(std::make_pair(0, std::move(t)));
    }
    return schedule;
}
} // namespace MNN
