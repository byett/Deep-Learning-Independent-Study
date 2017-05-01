require 'dp'
require 'optim'
require 'nn'
require 'torch'
cmd = torch.CmdLine()
cmd:text()
cmd:text('Image Classification using MLP Training/Optimization')
cmd:text('Example:')
cmd:text('$> th neuralnetwork.lua --batchSize 128 --momentum 0.5')
cmd:text('Options:')
--Most of these options are not particularly important for much of the time
cmd:option('--learningRate', 0.1, 'learning rate at t=0')
cmd:option('--lrDecay', 'linear', 'type of learning rate decay : adaptive | linear | schedule | none')
cmd:option('--minLR', 0.00001, 'minimum learning rate')
cmd:option('--saturateEpoch', 300, 'epoch at which linear decayed LR will reach minLR')
cmd:option('--schedule', '{}', 'learning rate schedule')
cmd:option('--maxWait', 4, 'maximum number of epochs to wait for a new minima to be found. After that, the learning rate is decayed by decayFactor.')
cmd:option('--decayFactor', 0.001, 'factor by which learning rate is decayed for adaptive decay.')
cmd:option('--maxOutNorm', 1, 'max norm each layers output neuron weights')
cmd:option('--momentum', 0, 'momentum')
--Additions I made to the example code attempting to increase the number of hidden layers. Not successful past 6 or 7 layers
cmd:option('--hiddenSize', '{200,200}', 'number of hidden units per layer')
cmd:option('--hiddenSize1', '{190,190}', 'number of hidden units per layer')
cmd:option('--hiddenSize2', '{180,180}', 'number of hidden units per layer')
cmd:option('--hiddenSize3', '{170,170}', 'number of hidden units per layer')
cmd:option('--hiddenSize4', '{160,160}', 'number of hidden units per layer')
cmd:option('--hiddenSize5', '{150,150}', 'number of hidden units per layer')
cmd:option('--hiddenSize6', '{140,140}', 'number of hidden units per layer')
cmd:option('--hiddenSize7', '{130,130}', 'number of hidden units per layer')
cmd:option('--hiddenSize8', '{120,120}', 'number of hidden units per layer')
cmd:option('--hiddenSize9', '{110,110}', 'number of hidden units per layer')
cmd:option('--hiddenSize10', '{100,100}', 'number of hidden units per layer')
cmd:option('--hiddenSize11', '{90,90}', 'number of hidden units per layer')
cmd:option('--hiddenSize12', '{80,80}', 'number of hidden units per layer')
cmd:option('--hiddenSize13', '{70,70}', 'number of hidden units per layer')
cmd:option('--hiddenSize14', '{60,60}', 'number of hidden units per layer')
--batchSize, maxEpoch, and dataset are the main parameters here
cmd:option('--batchSize', 20, 'number of examples per batch')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')
cmd:option('--maxEpoch', 10, 'maximum number of epochs to run')
cmd:option('--maxTries', 10, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--dropout', false, 'apply dropout on hidden neurons')
cmd:option('--batchNorm', false, 'use batch normalization. dropout is mostly redundant with this')
cmd:option('--dataset', 'Mnist', 'which dataset to use : Mnist | NotMnist | Cifar10 | Cifar100')
cmd:option('--standardize', false, 'apply Standardize preprocessing')
cmd:option('--zca', false, 'apply Zero-Component Analysis whitening')
cmd:option('--progress', false, 'display progress bar')
cmd:option('--silent', false, 'dont print anything to stdout')
cmd:text()
opt = cmd:parse(arg or {})
--Takes the defined values and repurposes them to actually be used in the program
opt.schedule = dp.returnString(opt.schedule)
opt.hiddenSize = dp.returnString(opt.hiddenSize)
opt.hiddenSize1 = dp.returnString(opt.hiddenSize1)
opt.hiddenSize2 = dp.returnString(opt.hiddenSize2)
opt.hiddenSize3 = dp.returnString(opt.hiddenSize3)
opt.hiddenSize4 = dp.returnString(opt.hiddenSize4)
opt.hiddenSize5 = dp.returnString(opt.hiddenSize5)
opt.hiddenSize6 = dp.returnString(opt.hiddenSize6)
opt.hiddenSize7 = dp.returnString(opt.hiddenSize7)
opt.hiddenSize8 = dp.returnString(opt.hiddenSize8)
opt.hiddenSize9 = dp.returnString(opt.hiddenSize9)
opt.hiddenSize10 = dp.returnString(opt.hiddenSize10)
opt.hiddenSize11 = dp.returnString(opt.hiddenSize11)
opt.hiddenSize12 = dp.returnString(opt.hiddenSize12)
opt.hiddenSize13 = dp.returnString(opt.hiddenSize13)
opt.hiddenSize14 = dp.returnString(opt.hiddenSize14)

--Haven't found this to be necessary, but the option is there
--[[preprocessing]]--
local input_preprocess = {}
if opt.standardize then
   table.insert(input_preprocess, dp.Standardize())
end
if opt.zca then
   table.insert(input_preprocess, dp.ZCA())
end
if opt.lecunlcn then
   table.insert(input_preprocess, dp.GCN())
   table.insert(input_preprocess, dp.LeCunLCN{progress=true})
end

--Loading in the chosen dataset
--[[data]]--
if opt.dataset == 'Mnist' then
   ds = dp.Mnist{input_preprocess = input_preprocess}
elseif opt.dataset == 'NotMnist' then
   ds = dp.NotMnist{input_preprocess = input_preprocess}
elseif opt.dataset == 'Cifar10' then
   ds = dp.Cifar10{input_preprocess = input_preprocess}
elseif opt.dataset == 'Cifar100' then
   ds = dp.Cifar100{input_preprocess = input_preprocess}
else
    error("Unknown Dataset")
end

--[[Model]]--
model = nn.Sequential() -- Name the model
model:add(nn.Convert(ds:ioShapes(), 'bf')) -- to batchSize x nFeature (also type converts). Needed because of potentially different image sizes

-- Hidden Layers. Each one of these takes in the inputSize and predefined hiddenSize (from initial parameters)
-- Also has an activation function between each layer before defining the new input size
-- Most of these are commented out for now
inputSize = ds:featureSize()
  for i,hiddenSize in ipairs(opt.hiddenSize) do
   model:add(nn.Linear(inputSize, hiddenSize)) -- parameters
   model:add(nn.ReLU())
   inputSize=hiddenSize
end
  for i,hiddenSize in ipairs(opt.hiddenSize1) do
   model:add(nn.Linear(inputSize, hiddenSize)) -- parameters
   model:add(nn.ReLU())
   inputSize=hiddenSize
end
  for i,hiddenSize in ipairs(opt.hiddenSize2) do
   model:add(nn.Linear(inputSize, hiddenSize)) -- parameters
   model:add(nn.ReLU())
   inputSize=hiddenSize
end
  for i,hiddenSize in ipairs(opt.hiddenSize3) do
   model:add(nn.Linear(inputSize, hiddenSize)) -- parameters
   model:add(nn.ReLU())
   inputSize=hiddenSize
end
  --[[for i,hiddenSize in ipairs(opt.hiddenSize4) do
   model:add(nn.Linear(inputSize, hiddenSize)) -- parameters
   model:add(nn.ReLU())
   inputSize=hiddenSize
end
  for i,hiddenSize in ipairs(opt.hiddenSize5) do
   model:add(nn.Linear(inputSize, hiddenSize)) -- parameters
   model:add(nn.ReLU())
   inputSize=hiddenSize
end
  for i,hiddenSize in ipairs(opt.hiddenSize6) do
   model:add(nn.Linear(inputSize, hiddenSize)) -- parameters
   model:add(nn.ReLU())
   inputSize=hiddenSize
end
  for i,hiddenSize in ipairs(opt.hiddenSize7) do
   model:add(nn.Linear(inputSize, hiddenSize)) -- parameters
   model:add(nn.ReLU())
   inputSize=hiddenSize
end
  for i,hiddenSize in ipairs(opt.hiddenSize8) do
   model:add(nn.Linear(inputSize, hiddenSize)) -- parameters
   model:add(nn.ReLU())
   inputSize=hiddenSize
end
  for i,hiddenSize in ipairs(opt.hiddenSize9) do
   model:add(nn.Linear(inputSize, hiddenSize)) -- parameters
   model:add(nn.ReLU())
   inputSize=hiddenSize
end
  for i,hiddenSize in ipairs(opt.hiddenSize10) do
   model:add(nn.Linear(inputSize, hiddenSize)) -- parameters
   model:add(nn.ReLU())
   inputSize=hiddenSize
end
  for i,hiddenSize in ipairs(opt.hiddenSize11) do
   model:add(nn.Linear(inputSize, hiddenSize)) -- parameters
   model:add(nn.ReLU())
   inputSize=hiddenSize
end
  for i,hiddenSize in ipairs(opt.hiddenSize12) do
   model:add(nn.Linear(inputSize, hiddenSize)) -- parameters
   model:add(nn.ReLU())
   inputSize=hiddenSize
end
  for i,hiddenSize in ipairs(opt.hiddenSize13) do
   model:add(nn.Linear(inputSize, hiddenSize)) -- parameters
   model:add(nn.ReLU())
   inputSize=hiddenSize
end
  for i,hiddenSize in ipairs(opt.hiddenSize14) do
   model:add(nn.Linear(inputSize, hiddenSize)) -- parameters
   model:add(nn.ReLU())
   inputSize=hiddenSize
end]]--

-- output layer
model:add(nn.Linear(inputSize, #(ds:classes()))) -- Input is the output size of previous layer, output is number of classes we are sorting into. For MNIST, that would be 10.
model:add(nn.LogSoftMax())

--[[Propagators]]--
if opt.lrDecay == 'adaptive' then
   ad = dp.AdaptiveDecay{max_wait = opt.maxWait, decay_factor=opt.decayFactor}
elseif opt.lrDecay == 'linear' then
   opt.decayFactor = (opt.minLR - opt.learningRate)/opt.saturateEpoch
end

train = dp.Optimizer{ -- Create the trainer for the model. Different options can be triggered from the initial parameters
   acc_update = opt.accUpdate,
   loss = nn.ModuleCriterion(nn.ClassNLLCriterion(), nil, nn.Convert()),
   epoch_callback = function(model, report) -- called every epoch
      -- learning rate decay
      if report.epoch > 0 then
         if opt.lrDecay == 'adaptive' then
            opt.learningRate = opt.learningRate*ad.decay
            ad.decay = 1
         elseif opt.lrDecay == 'schedule' and opt.schedule[report.epoch] then
            opt.learningRate = opt.schedule[report.epoch]
         elseif opt.lrDecay == 'linear' then 
            opt.learningRate = opt.learningRate + opt.decayFactor
         end
         opt.learningRate = math.max(opt.minLR, opt.learningRate)
         if not opt.silent then
            print("learningRate", opt.learningRate)
         end
      end
   end,
   callback = function(model, report) -- called every batch
      if opt.accUpdate then
         model:accUpdateGradParameters(model.dpnn_input, model.output, opt.learningRate)
      else
         model:updateGradParameters(opt.momentum) -- affects gradParams
         model:updateParameters(opt.learningRate) -- affects params
      end
      model:maxParamNorm(opt.maxOutNorm) -- affects params
      model:zeroGradParameters() -- affects gradParams 
   end,
   feedback = dp.Confusion(),
   sampler = dp.ShuffleSampler{batch_size = opt.batchSize},
   progress = opt.progress
}
valid = dp.Evaluator{ -- Create the validation aspect of the model
   feedback = dp.Confusion(),  
   sampler = dp.Sampler{batch_size = opt.batchSize}
}
test = dp.Evaluator{ -- Create the testing aspect of the model
   feedback = dp.Confusion(),
   sampler = dp.Sampler{batch_size = opt.batchSize}
}

--[[Experiment]]--
xp = dp.Experiment{
   model = model,
   optimizer = train,
   validator = valid,
   tester = test,
   observer = {
      dp.FileLogger(),
      dp.EarlyStopper{
         error_report = {'validator','feedback','confusion','accuracy'},
         maximize = true,
         max_epochs = opt.maxTries
      }
   },
   random_seed = os.time(),
   max_epoch = opt.maxEpoch -- How many times the experiment runs; defined previously
}