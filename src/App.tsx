import { ChangeEvent, useState } from 'react'
import { SSDMobileNetV1, TinyYOLOv2 } from './utils/object-detections'
import { streamInferenceOD } from './utils/stream-inference-wrapper'
import { onloadmetadataPromise } from './utils/video-util'
import { RenderBoundingBoxToCanvasStream } from './utils/image-util'
import { MiddlewareStream } from './utils/middleware-stream'
import { FPSCounter } from './utils/fps-counter'
import { Webcam } from './utils/webcam'

const models = ['tinyyolov2', 'ssdmobilenetv1']
const providers: Provider[] = ['cpu', 'gpu']

const getModel = (modelName: string, provider: Provider) => {
  switch (modelName) {
    case 'tinyyolov2':
      return new TinyYOLOv2(provider, 0.3)
    case 'ssdmobilenetv1':
      return new SSDMobileNetV1(provider, 0.3)
    default:
      return new TinyYOLOv2(provider, 0.3)
  }
}

const App = () => {
  const [controller, setController] = useState<AbortController>()
  const [currentModel, setCurrentModel] = useState<string>(models[0])
  const [currentProvider, setCurrentProvideer] = useState<Provider>(
    providers[0],
  )
  const [fps, setFps] = useState('FPS')
  const onClickStart = () => {
    const start = async () => {
      try {
        console.log('Start streaming...')
        const cameraStream = await Webcam.start()
        const video = document.getElementById('video') as HTMLVideoElement
        video.srcObject = cameraStream
        video.play()
        await onloadmetadataPromise(video)
        const videoSize = {
          width: video.videoWidth,
          height: video.videoHeight,
        }
        const model = getModel(currentModel, currentProvider)
        await model.initialize()
        const controller = new AbortController()
        setController(controller)
        const inferenceStream = streamInferenceOD(
          Webcam.imageCapture,
          model,
          controller.signal,
        )
        const canvas = document.getElementById('canvas') as HTMLCanvasElement
        canvas.width = videoSize.width
        canvas.height = videoSize.height
        const canvasRenderer = new RenderBoundingBoxToCanvasStream(
          canvas.getContext('2d')!,
          videoSize,
          'green',
        )
        const fpsCounter = new FPSCounter(10, setFps)
        const fpsCounterMiddleware = new MiddlewareStream(fpsCounter.click)
        await inferenceStream
          .pipeThrough(fpsCounterMiddleware)
          .pipeTo(canvasRenderer)
      } catch (error) {
        console.error(error)
      } finally {
        Webcam.stop()
      }
    }
    start()
  }

  const onChangeModel = (event: ChangeEvent<HTMLInputElement>) => {
    setCurrentModel(event.target.value)
  }

  const onChangeProvider = (event: ChangeEvent<HTMLInputElement>) => {
    setCurrentProvideer(event.target.value as Provider)
  }

  return (
    <div className="App">
      <div onChange={onChangeModel}>
        {models.map((model) => {
          return (
            <>
              <input
                type="radio"
                value={model}
                name="model"
                checked={model === currentModel}
              />{' '}
              {model}
            </>
          )
        })}
      </div>
      <div onChange={onChangeProvider}>
        {providers.map((provider) => {
          return (
            <>
              <input
                type="radio"
                value={provider}
                name="provider"
                checked={provider === currentProvider}
              />{' '}
              {provider}
            </>
          )
        })}
      </div>
      <div>
        <button onClick={onClickStart}>Start</button>
        <button
          onClick={() => {
            if (controller) {
              controller.abort()
            }
          }}
        >
          Stop
        </button>
        <p>{fps}</p>
      </div>
      <div style={{ position: 'relative', zIndex: 0 }}>
        <video
          id="video"
          style={{ position: 'absolute', top: 0, left: 0 }}
        ></video>
        <canvas
          id="canvas"
          style={{ position: 'relative', top: 0, left: 0, zIndex: 1 }}
        ></canvas>
      </div>
    </div>
  )
}

export default App
