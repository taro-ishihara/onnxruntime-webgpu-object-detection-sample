import { ObjectDetection } from './object-detections'

const streamInferenceOD = (
  imageCapture: ImageCapture,
  model: ObjectDetection,
  signal: AbortSignal,
) => {
  const stream = new ReadableStream({
    start(controller) {
      async function pushFrame() {
        if (signal.aborted) {
          controller.close()
          return
        }
        const image: ImageBitmap = await imageCapture.grabFrame()
        const input = model.preprocess(image)
        const output = await model.inference(input)
        const boundingBox = model.postprocess(output)
        controller.enqueue(boundingBox)
        requestAnimationFrame(pushFrame)
      }
      pushFrame()
    },
  })
  return stream
}

export { streamInferenceOD }
