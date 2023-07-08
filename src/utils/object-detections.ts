import { InferenceSession, TypedTensor, Tensor } from 'onnxruntime-web'
import { InferenceSession as InferenceSessionGPU } from 'onnxruntime-web/webgpu'
import { getResizedRGBDataFromBitmap } from './image-util'
import { tensor3d } from '@tensorflow/tfjs'
import { YoloOutputParser } from './yolo-output-parser'

abstract class ObjectDetection {
  protected _modelPath: string
  protected _inputSize: ImageSize
  protected _provider: Provider
  protected _threshold: number
  protected _session: InferenceSession | InferenceSessionGPU | null

  constructor(
    modelPath: string,
    inputSize: ImageSize,
    provider: Provider,
    threshold: number,
  ) {
    this._modelPath = modelPath
    this._inputSize = inputSize
    this._provider = provider
    this._threshold = threshold
    this._session = null
  }

  get threshold(): number {
    return this._threshold
  }

  set threshold(value: number) {
    this._threshold = value
  }

  public initialize = async () => {
    switch (this._provider) {
      case 'cpu': {
        this._session = await InferenceSession.create(this._modelPath, {
          executionProviders: ['wasm'],
        })
        break
      }
      case 'gpu': {
        this._session = await InferenceSessionGPU.create(this._modelPath, {
          executionProviders: ['webgpu'],
        })
        break
      }
      default: {
        throw Error('Provider is not supported.')
      }
    }
    console.log(
      `Initialized ${this._provider} session: ${
        Object.getPrototypeOf(this).constructor.name
      }`,
    )
  }

  abstract preprocess(image: ImageBitmap): TypedTensor<Tensor.Type>
  abstract inference(
    inputTensor: TypedTensor<Tensor.Type>,
  ): Promise<any> | undefined
  abstract postprocess(output: any): BoundingBox[]
}

class TinyYOLOv2 extends ObjectDetection {
  constructor(provider: Provider, threshold: number) {
    super(
      './static/models/tinyyolov2-8.onnx',
      { width: 416, height: 416 },
      provider,
      threshold,
    )
  }

  private createInputTensorFromRGBData = (rgbData: Uint8ClampedArray) => {
    const tfTensor4d = tensor3d(
      new Float32Array(rgbData),
      [this._inputSize.height, this._inputSize.width, 3],
      'float32',
    )
      .transpose([2, 0, 1])
      .expandDims()
    const chwData = tfTensor4d.dataSync()
    tfTensor4d.dispose()
    const inputTensor = new Tensor(chwData, [
      1,
      3,
      this._inputSize.width,
      this._inputSize.height,
    ])
    return inputTensor
  }

  public preprocess = (image: ImageBitmap): TypedTensor<Tensor.Type> => {
    const rgbData = getResizedRGBDataFromBitmap(image, this._inputSize)
    const inputTensor = this.createInputTensorFromRGBData(rgbData)
    return inputTensor
  }

  public inference = (
    inputTensor: TypedTensor<Tensor.Type>,
  ): Promise<any> | undefined => {
    return this._session?.run({
      [this._session.inputNames[0]]: inputTensor,
    })
  }

  public postprocess = (output: any): BoundingBox[] => {
    const parsedOutput = YoloOutputParser.parseOutputs(output.grid.data, 0.3)
    const filteredOutput = YoloOutputParser.filterBoundingBoxes(
      parsedOutput,
      5,
      this._threshold,
    )
    return filteredOutput
      .filter((value) => value.label === 14)
      .map((value) => {
        return {
          x: value.boundingBox.x / this._inputSize.width,
          y: value.boundingBox.y / this._inputSize.height,
          width: value.boundingBox.width / this._inputSize.width,
          height: value.boundingBox.height / this._inputSize.height,
        }
      })
  }
}

class SSDMobileNetV1 extends ObjectDetection {
  constructor(provider: Provider, threshold: number) {
    super(
      './static/models/ssd_mobilenet_v1_10.onnx',
      { width: 224, height: 224 },
      provider,
      threshold,
    )
  }

  private createInputTensorFromRGBData = (rgbData: Uint8ClampedArray) => {
    return new Tensor(new Uint8Array(rgbData), [
      1,
      this._inputSize.height,
      this._inputSize.width,
      3,
    ])
  }

  private arrayToBoundingBox = (array: Float32Array): BoundingBox => {
    const top = array[0]
    const left = array[1]
    const height = array[2] - top
    const width = array[3] - left
    return { x: left, y: top, width: width, height: height }
  }

  public preprocess = (image: ImageBitmap): TypedTensor<Tensor.Type> => {
    const rgbData = getResizedRGBDataFromBitmap(image, this._inputSize)
    const inputTensor = this.createInputTensorFromRGBData(rgbData)
    return inputTensor
  }

  public inference = (
    inputTensor: TypedTensor<Tensor.Type>,
  ): Promise<any> | undefined => {
    return this._session?.run({ [this._session.inputNames[0]]: inputTensor })
  }

  public postprocess = (output: any): BoundingBox[] => {
    const boxes = output[this._session!.outputNames[0]].data as Float32Array
    const classes = output[this._session!.outputNames[1]].data as Float32Array
    const scores = output[this._session!.outputNames[2]].data as Float32Array

    const result: BoundingBox[] = []
    for (let i = 0; i < scores.length; i++) {
      if (scores[i] > this.threshold && classes[i] === 1) {
        const start = i * 4
        const end = start + 4
        const box = this.arrayToBoundingBox(boxes.slice(start, end))
        result.push(box)
      } else {
        break
      }
    }
    return result
  }
}

export { SSDMobileNetV1, TinyYOLOv2, ObjectDetection }
