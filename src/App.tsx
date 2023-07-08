import { ChangeEvent, useState } from 'react'
import { SSDMobileNetV1, TinyYOLOv2 } from './utils/object-detections'
import { streamInferenceOD } from './utils/stream-inference-wrapper'
import { onloadmetadataPromise } from './utils/video-util'
import { RenderBoundingBoxToCanvasStream } from './utils/image-util'
import { MiddlewareStream } from './utils/middleware-stream'
import { FPSCounter } from './utils/fps-counter'
import { Webcam } from './utils/webcam'

const App = () => {
  const [controller, setController] = useState<AbortController>()
  const [provider, setProvideer] = useState<Provider>('cpu')
  const [fps, setFps] = useState('')
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
        const model = new TinyYOLOv2(provider, 0.3)
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
        const fpsCounter = new FPSCounter(30, setFps)
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

  const onChangeProvider = (event: ChangeEvent<HTMLInputElement>) => {
    setProvideer(event.target.value as Provider)
  }

  return (
    <div className="App">
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
      <div onChange={onChangeProvider}>
        <input type="radio" value="cpu" name="provider" /> CPU
        <input type="radio" value="gpu" name="provider" /> GPU
      </div>
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
  )
}

export default App
