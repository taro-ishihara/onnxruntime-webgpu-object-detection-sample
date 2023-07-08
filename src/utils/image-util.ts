const getResizedRGBDataFromBitmap = (
  image: ImageBitmap,
  newSize: ImageSize,
) => {
  const canvas = document.createElement('canvas')
  canvas.width = newSize.width
  canvas.height = newSize.height
  const context = canvas.getContext('2d')!
  context.drawImage(
    image,
    0,
    0,
    image.width,
    image.height,
    0,
    0,
    newSize.width,
    newSize.height,
  )
  const imageData = context.getImageData(0, 0, newSize.width, newSize.height)

  const pixelData = imageData.data
  const rgbData = pixelData.filter((value, index) => index % 4 !== 3)

  return rgbData
}

const renderBoundingBoxToCanvas = (
  context: CanvasRenderingContext2D,
  boxes: Array<BoundingBox>,
  size: ImageSize,
  color: string,
) => {
  context.strokeStyle = color
  context.clearRect(0, 0, size.width, size.height)
  boxes.forEach((box) => {
    context.strokeRect(
      box.x * size.width,
      box.y * size.height,
      box.width * size.width,
      box.height * size.height,
    )
  })
}

class RenderBoundingBoxToCanvasStream extends WritableStream<any> {
  constructor(
    context: CanvasRenderingContext2D,
    size: ImageSize,
    color: string,
  ) {
    super({
      write(chunk: BoundingBox[]) {
        renderBoundingBoxToCanvas(context, chunk, size, color)
      },
      close() {
        console.log('Stream ended.')
      },
    })
  }
}

export {
  getResizedRGBDataFromBitmap,
  renderBoundingBoxToCanvas,
  RenderBoundingBoxToCanvasStream,
}
