# TIDL-阅读笔记 
### 常用结构体
```
typedef struct
{
  IVISION_Params visionParams;
  sTIDL_Network_t net;
  int32_t currCoreId;
  int32_t currLayersGroupId;
  int32_t l1MemSize;
  int32_t l2MemSize;
  int32_t l3MemSize;
  int32_t quantHistoryParam1;
  int32_t quantHistoryParam2;
  int32_t quantMargin;
  int32_t optimiseExtMem;
} TIDL_CreateParams;
```
TIDL_CreateParams 存储了网络模型，以及内存参数，还有一些量化的数据，以后会大量使用到。以下是其依赖的一些结构体。
```
typedef struct IVISION_Params {
    IALG_Params algParams;       /**< IALG Params */
    ivisionCacheWriteBack cacheWriteBack; /**< Function pointer for cache
                                          write back for cached based system.
                                          If the system is not using cache for
                                          data memory then the pointer can be
                                          filled with NULL. If the algorithm recives
                                          a input buffer with IVISION_AccessMode as
                                          @sa IVISION_ACCESSMODE_CPU and the
                                          @sa ivisionCacheWriteBack as NULL then
                                          the algorithm will return with NULL
                                          */
} IVISION_Params;

typedef struct {
  int32_t numLayers;
  int32_t weightsElementSize;
  int32_t slopeElementSize;
  int32_t biasElementSize;
  int32_t dataElementSize;
  int32_t interElementSize;
  int32_t quantizationStyle;
  int32_t strideOffsetMethod;
  int32_t reserved;
  sTIDL_Layer_t TIDLLayers[TIDL_NUM_MAX_LAYERS];
}sTIDL_Network_t;

typedef struct {
  sTIDL_LayerParams_t layerParams;
  int32_t layerType;
  int32_t numInBufs;
  int32_t numOutBufs;
  sTIDL_DataParams_t inData[TIDL_NUM_IN_BUFS]; 
  sTIDL_DataParams_t outData[TIDL_NUM_OUT_BUFS];
  int32_t coreID;
  int32_t layersGroupId; 
  int32_t weightsElementSizeInBits;
}sTIDL_Layer_t;

typedef union {
  sTIDL_ConvParams_t              convParams;
  sTIDL_ReLUParams_t              reluParams;
  sTIDL_EltWiseParams_t           eltWiseParams;
  sTIDL_PoolingParams_t           poolParams;
  sTIDL_InnerProductParams_t      innerProductParams;
  sTIDL_DataLayerParams_t         dataLayerParams;
  sTIDL_ArgMaxParams_t            argMaxParams;
  sTIDL_SoftMaxParams_t           softMaxParams;
  sTIDL_BiasParams_t              biasParams;   
  sTIDL_BatchNormParams_t         batchNormParams;   
}sTIDL_LayerParams_t;


typedef struct {
  sBuffer_t weights;
  sBuffer_t bias;
  int32_t   convolutionType;
  int32_t   numInChannels;
  int32_t   numOutChannels;
  int32_t   numGroups;
  int32_t   kernelW;
  int32_t   kernelH;
  int32_t   strideW;
  int32_t   strideH;
  int32_t   dilationW;
  int32_t   dilationH;
  int32_t   padW;
  int32_t   padH;
  int32_t   weightsQ;
  int32_t   zeroWeightValue;
  int32_t   biasQ;
  int32_t   inDataQ;
  int32_t   outDataQ;
  int32_t   interDataQ;
  int32_t   enableBias;
  int32_t   enablePooling;
  int32_t   enableRelU;
  int32_t   kernelType;  
  sTIDL_PoolingParams_t poolParams;
  sTIDL_ReLUParams_t    reluParams;
}sTIDL_ConvParams_t;

typedef struct {
  int32_t dataId;
  int32_t elementType;    
  int32_t numDim;
  int32_t dataQ;
  int32_t minValue;
  int32_t maxValue;
  int32_t pitch[TIDL_DIM_MAX-1];
  int32_t dimValues[TIDL_DIM_MAX];
}sTIDL_DataParams_t;

```
tidl_tb.c这个文件时TIDL PC模拟端的入口文件，里面定义了main函数。main函数流程：
- 解析配置文件，存入```tidl_conv2d_config gParams```
- 进行模型测试：```status = test_ti_dl_ivison();```

test_ti_dl_ivison函数流程：
- 分配内存空间，L1，L2，L3；
>```
>#if CORE_DSP
>#define DMEM0_SIZE (8*1024)
>#define DMEM1_SIZE (148*1024)
>#else
>#define DMEM0_SIZE (20*1024)
>#define DMEM1_SIZE (8*1024)
>#endif
>#define OCMC_SIZE  (320*1024)
>```
然后申请网络参数空间，以及填充空间
```
tidl_allocNetParamsMem(&createParams.net);
tidl_fillNetParamsMem(&createParams.net,params);
```
- tidl_allocNetParamsMem:计算各个层的参数数量，计算每层的bufSize，给所有参数的ptr分配空间，从memObj_EXTMEMNONCACHEIO地址申请空间
- tidltb_printNetInfo:打印网络数据
- numMemRec = TIDL_VISION_FXNS.ialg.algNumAlloc();//9大内存
  memRec    = (IALG_MemRec *)malloc(numMemRec*sizeof(IALG_MemRec));//sizeof(IALG_MemRec) = 20,what's the meaning?
- status = TIDL_VISION_FXNS.ialg.algAlloc(
    (IALG_Params *)(&createParams), NULL, memRec);//Alloc内存
### 九大内存使用说明：

typedef enum
{
  /* Memory records for handle */
  ALG_HANDLE_MEMREC,
  ALG_HANDLE_INT_MEMREC,
  ALG_CREATE_PARAM_MEMREC,
  ALG_SCRATCH_L1_MEM_MEMREC,
  ALG_SCRATCH_L2_MEM_MEMREC,
  ALG_SCRATCH_L3_MEM_MEMREC,
  ALG_LAYERS_PARAMS_BUFF_MEMREC,
  ALG_SCRATCH_DATA_BUFF_MEMREC,
  ALG_LAYERS_MEMREC,
  MAX_NUM_MEMRECS
} eMemrecs;
 
| MEM_IDX | DESCRIPTION |
| ------ | ------ |
| ALG_HANDLE_MEMREC  | 存储TIDL_OBJ - IALG_EXTERNAL - sizeof(TIDL_Obj)这个memrec在运行的时候并不执行|
| ALG_HANDLE_INT_MEMREC | 存储TIDL_OBJ初始化后的备份  activate的后手从ALG_HANDLE_MEMREC拷贝至此  进行process时调用参数 - IALG_DARAM0 - ALIGN_SIZE(sizeof(TIDL_Obj),8)|
| ALG_SCRATCH_L1_MEM_MEMREC | 并没有存储 用于实时计算时的缓冲加速 - IALG_DARAM0 - ALIGN_SIZE((createParam->l1MemSize  - ALIGN_SIZE(sizeof(TIDL_Obj),128) - 256), 128)|
| ALG_SCRATCH_L2_MEM_MEMREC | 并没有存储 用于实时计算时的缓冲加速 - IALG_DARAM1 - createParam->l2MemSize - 128
| ALG_SCRATCH_L3_MEM_MEMREC | 并没有存储 用于实时计算时的缓冲加速 - IALG_SARAM0 - createParam->l3MemSize
| ALG_CREATE_PARAM_MEMREC | 存储每个model的createParam --- IALG_EXTERNAL --- sizeof(TIDL_CreateParams)|
| ALG_LAYERS_PARAMS_BUFF_MEMREC | 存储卷积层的网络参数（该参数是由createParam里的ptr解析转化而来）- IALG_EXTERNAL - 根据卷积层的配置而定|
| ALG_LAYERS_MEMREC | 存储sTIDL_LayerBuf_t(每层的outSize、indataid、outdataid) - IALG_EXTERNAL - ALIGN_SIZE(sizeof(sTIDL_LayerBuf_t), 128)|
| ALG_SCRATCH_DATA_BUFF_MEMREC| 网络的outbuffer是放入这个区域的|

external memory 和 internal memory的内存分布是怎样的？ 
Tda2x的实际内存布局是怎样的？
```
/**
*  @enum   esysMemScratch
*  @brief  Memory records for scratch memories
*/
typedef enum
{
  TIDL_SYSMEM_L1_SCRATCH,
  TIDL_SYSMEM_L2_SCRATCH,
  TIDL_SYSMEM_L3_SCRATCH,
  TIDL_SYSMEM_IBUFL,
  TIDL_SYSMEM_IBUFH,
  TIDL_SYSMEM_WBUF,
  TIDL_SYSMEM_MAX
} esysMemScratch;
```
TIDL_conv2dBlockProps函数的实现原理，不懂...
ALIGN_SIZE（）原理？
以下结构体是TI描述内存的ptrBase，ptrCurr，ptrTotalSize，ptrAvailSize
```
typedef struct _memory
{
  unsigned char *ptrBase;
  unsigned char *ptrCurr;
  unsigned int  u32Totalsize;
  unsigned int  u32AvailableSize;
} sMemory_t ;
```

TestApp_AllocMemRecords函数进行内存alloc，从memObj_DMEM0和memObj_DMEM1内部空间申请空间,将地址赋值给memRec[i].base

```
if(createParams->optimiseExtMem != TIDL_optimiseExtMemL0){}这个优化的逻辑，不太理解
```
### 三大函数
- TIDL_alloc
对9大内存的属性(size, space, attrs, alignment)等进行赋值，调用```TIDL_conv2DAlloc```等各网络层函数对各层进行alloc，根据参数量增加九大内存中对应分区的size。
```
memRec[ALG_LAYERS_PARAMS_BUFF_MEMREC].size += (numCoeffBuffer + 128U);
memRec[ALG_LAYERS_PARAMS_BUFF_MEMREC].size += (coeffBuffer + 128U);
memRec[ALG_LAYERS_PARAMS_BUFF_MEMREC].size += (offsetBuffer + 128U);
memRec[ALG_SCRATCH_DATA_BUFF_MEMREC].size  += (scratchDataSize + 128U);
```

- TIDL_init
函数入口：```status = TIDL_VISION_FXNS.ialg.algInit((IALG_Handle)(&handle),
    memRec,NULL,(IALG_Params *)(&createParams));```
调用各层自己的init函数进行初始化，conv：
```
static XDAS_Int32 TIDL_conv2DInit(const TIDL_CreateParams *params, 
int32_t layerIdx,
sTIDL_AlgLayer_t * algLayer, int32_t *paramMemTabOffset,
int32_t *dataMemTabOffset, const IALG_MemRec memRec[], void ** outPtr,
int32_t quantizationStyle, sTIDL_LayerBuf_t *TIDLLayersBuf){}
```
- TIDL_process
函数入口：```status = handle->ivision->algProcess((IVISION_Handle)handle,
      &inBufs,&outBufs,(IVISION_InArgs *)&inArgs,(IVISION_OutArgs *)&outArgs);```
```
/**
 @struct  sTIDL_DataParams_t
 @brief   This structure define the parmeters of data or kerner buffer
          used by TIDL layers (In,Out)
 @param  dataId
          Address pointing to the actual buffer
 @param  elementType
          Size of the buffer in bytes
 @param  numDim
          Address pointing to the actual buffer
 @param  dataQ
          Number of bits for fractional part if Quant Style is 1
          Q factor if Quant Style is 2
 @param  minValue
          Minimum value of 32-bit accumulator for all the values in 
					that layer	
 @param  maxValue
          Maximum value of 32-bit accumulator for all the values in 
					that layer	
 @param  pitch
          Pitch for each dimention
 @param  dimValues
          Size of the buffer in bytes

*/
typedef struct {
  int32_t dataId;
  int32_t elementType;    
  int32_t numDim;
  int32_t dataQ;
  int32_t minValue;
  int32_t maxValue;
  int32_t pitch[TIDL_DIM_MAX-1];
  int32_t dimValues[TIDL_DIM_MAX];
}sTIDL_DataParams_t;
```
用```sTIDL_DataParams_t```来存储inDataParams和outDataParams
程序中的数据都是以```dataId```的形式获取的，这个dataId是在哪儿索引的呢？

```
/**
 @enum    eTIDL_DataDimIndex
 @brief   This enumerator defines the indices of dimension array of layer data
          buffer in TIDL library
*/
typedef enum
{
  TIDL_DIM_BATCH          = 0,
  TIDL_DIM_NUMCH          = 1,
  TIDL_DIM_HEIGHT         = 2,
  TIDL_DIM_WIDTH          = 3,
  TIDL_DIM_MAX            = 4       
}eTIDL_DataDimIndex;

/**
 @enum    eTIDL_PitchDimIndex
 @brief   This enumerator defines the indices of picth array of layer data
          buffer in TIDL library
*/
typedef enum
{
  TIDL_ROI_PITCH         = 0,
  TIDL_CHANNEL_PITCH     = 1,
  TIDL_LINE_PITCH        = 2,
  TIDL_PITCH_MAX         = (TIDL_DIM_MAX - 1)
}eTIDL_PitchDimIndex;
```
```eTIDL_PitchDimIndex```是```pitch```的结构体
###### 流程
1) TIDL_activate((IALG_Handle)(void*)Handle);将ALG_HANDLE_MEMREC的内存通过DMA拷贝到ALG_HANDLE_INT_MEMREC中。
2) 然后进行sysMems内存的赋值
3) 循环处理每层结果：
    处理inPtrs和outPtrs
    调用各层对应的process函数，比如conv的```TIDL_conv2dProcess```
    
4) TIDL_conv2dProcess分析
    - 先进行网络参数初始化
    - 量化系数确定：
        - qFact = (params->inDataQ + 127)/256;//inDataQ大于127，那么qFace>=1；否则为0，后期被赋予给`buffParams.biasQFact = qFact;`
        - zeroWeightValue = params->zeroWeightValue;
    有两次量化过程：`buffParams.firstRoundBits  = roundBits;`和`buffParams.secondRoundBits = outRoundBits;`第一次量化是根据之前的outbuf中min,max来定义的
    - 调用TIDL_conv2dBlockProcess进行卷积运算。
        - TIDL_conv2dAllocIntMem，进行dma src和sink地址绑定
        - EDMA_UTILS_globalReset();
        - TIDL_conv2dDmaSrcInit
            ...
            status = TIDL_conv2dDmaSrcAutoIncrementInit(tidlDmaSrcPtr); ...
            status = TIDL_conv2dDmaBiasAutoIncrementInit(tidlDmaSrcPtr, biasElementSize);
            ...
        - TIDL_conv2dComputeInit
            ...  TIDL_initConv2dParamsRingBuff(tidlComputePtr); ...
        - TIDL_conv2dDmaSinkIni
            ...
        - DMA的具体操作，根据不同的硬件平台，代码执行也不一样。PC模拟是进入到等待DMA的过程：
            while(srcDMAStatus != -1){
            }
    - TIDL_StoreMinMax(min, max); 存储网络min max
    - 更新min max：
        `params->outDataQ = TIDL_updateMinMax(tidlLayer, params->inDataQ, params->weightsQ, 
											intAlgHandle->createParams->net.quantizationStyle,
											outPutShift, intAlgHandle->procCallCounter, min, max,
                            intAlgHandle->createParams->quantHistoryParam1,
                            intAlgHandle->createParams->quantHistoryParam2
                            );`
    - 如果定义了和ref对比的话：进行refConv计算，并且做对比。
        #if ENABLE_REF_COMPARISION
          status =TIDL_refConv2dProcess(
              intAlgHandle,
              algLayer,
              tidlLayer,
              params,
              &buffParams,
              (int8_t *)inPtr,
              (uint8_t *)outPtr,
              buffParams.inElementType,
              numTotRoi,
              sysMems);
