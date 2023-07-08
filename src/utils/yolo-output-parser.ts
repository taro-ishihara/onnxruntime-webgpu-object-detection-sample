class YoloOutputParser {
  private static readonly ROW_COUNT = 13
  private static readonly COL_COUNT = 13
  private static readonly BOXES_PER_CELL = 5
  private static readonly BOX_INFO_FEATURE_COUNT = 5
  private static readonly CLASS_COUNT = 20
  private static readonly CELL_WIDTH = 32
  private static readonly CELL_HEIGHT = 32
  private static readonly channelStride = this.ROW_COUNT * this.COL_COUNT

  private constructor() {}

  private static readonly anchors = [
    1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52,
  ]

  private static sigmoid(value: number): number {
    const k = Math.exp(value)
    return k / (1.0 + k)
  }

  private static softmax(values: number[]): number[] {
    const maxValue = Math.max(...values)
    const exp = values.map((value) => Math.exp(value - maxValue))
    const sumExp = exp.reduce(
      (accumulator, currentValue) => accumulator + currentValue,
      0,
    )

    return exp.map((value) => value / sumExp)
  }

  private static getOffset(x: number, y: number, channel: number): number {
    return channel * this.channelStride + y * this.COL_COUNT + x
  }

  private static extractBoundingBoxDimensions(
    modelOutput: Float32Array,
    x: number,
    y: number,
    channel: number,
  ): BoundingBox {
    return {
      x: modelOutput[this.getOffset(x, y, channel)],
      y: modelOutput[this.getOffset(x, y, channel + 1)],
      width: modelOutput[this.getOffset(x, y, channel + 2)],
      height: modelOutput[this.getOffset(x, y, channel + 3)],
    }
  }

  private static getConfidence(
    modelOutput: Float32Array,
    x: number,
    y: number,
    channel: number,
  ): number {
    return this.sigmoid(modelOutput[this.getOffset(x, y, channel + 4)])
  }

  private static mapBoundingBoxToCell(
    x: number,
    y: number,
    anchorBox: number,
    boundingBox: BoundingBox,
  ): BoundingBox {
    return {
      x: (x + this.sigmoid(boundingBox.x)) * this.CELL_WIDTH,
      y: (y + this.sigmoid(boundingBox.y)) * this.CELL_HEIGHT,
      width:
        Math.exp(boundingBox.width) *
        this.CELL_WIDTH *
        this.anchors[anchorBox * 2],
      height:
        Math.exp(boundingBox.height) *
        this.CELL_HEIGHT *
        this.anchors[anchorBox * 2 + 1],
    }
  }

  public static extractClasses(
    modelOutput: Float32Array,
    x: number,
    y: number,
    channel: number,
  ): number[] {
    const predictedClasses: number[] = Array.from({ length: this.CLASS_COUNT })
    const predictedClassOffset = channel + this.BOX_INFO_FEATURE_COUNT
    for (
      let predictedClass = 0;
      predictedClass < this.CLASS_COUNT;
      predictedClass++
    ) {
      predictedClasses[predictedClass] =
        modelOutput[this.getOffset(x, y, predictedClass + predictedClassOffset)]
    }
    return this.softmax(predictedClasses)
  }

  private static getTopResult(predictedClasses: number[]) {
    const topResult = predictedClasses
      .map((predictedClass, index) => {
        return { index: index, value: predictedClass }
      })
      .sort((a, b) => b.value - a.value)
      .find(() => true)!
    return topResult
  }

  private static intersectionOverUnion(
    boundingBoxA: BoundingBox,
    boundingBoxB: BoundingBox,
  ): number {
    const areaA = boundingBoxA.width * boundingBoxA.height
    if (areaA <= 0) {
      return 0
    }
    const areaB = boundingBoxB.width * boundingBoxB.height
    if (areaB <= 0) {
      return 0
    }
    const minX = Math.max(boundingBoxA.x, boundingBoxB.x)
    const minY = Math.max(boundingBoxA.y, boundingBoxB.y)
    const maxX = Math.min(
      boundingBoxA.x + boundingBoxA.width,
      boundingBoxB.x + boundingBoxB.width,
    )
    const maxY = Math.min(
      boundingBoxA.y + boundingBoxA.height,
      boundingBoxB.y + boundingBoxB.height,
    )

    const intersectionArea = Math.max(maxY - minY, 0) * Math.max(maxX - minX, 0)

    return intersectionArea / (areaA + areaB - intersectionArea)
  }

  public static parseOutputs(
    yoloModelOutputs: Float32Array,
    threshold: number,
  ): Detection[] {
    const detections = []
    for (let row = 0; row < this.ROW_COUNT; row++) {
      for (let column = 0; column < this.COL_COUNT; column++) {
        for (let anchorBox = 0; anchorBox < this.BOXES_PER_CELL; anchorBox++) {
          const channel =
            anchorBox * (this.CLASS_COUNT + this.BOX_INFO_FEATURE_COUNT)
          const boundingBoxDimensions = this.extractBoundingBoxDimensions(
            yoloModelOutputs,
            row,
            column,
            channel,
          )
          const confidence = this.getConfidence(
            yoloModelOutputs,
            row,
            column,
            channel,
          )
          const mappedBoundingBox = this.mapBoundingBoxToCell(
            row,
            column,
            anchorBox,
            boundingBoxDimensions,
          )
          if (confidence < threshold) {
            continue
          }
          const predictedClasses = this.extractClasses(
            yoloModelOutputs,
            row,
            column,
            channel,
          )
          const topResultScore = this.getTopResult(predictedClasses)
          const topScore = topResultScore.value * confidence
          if (topScore < threshold) {
            continue
          }
          const detection: Detection = {
            boundingBox: {
              x: mappedBoundingBox.x - mappedBoundingBox.width / 2,
              y: mappedBoundingBox.y - mappedBoundingBox.height / 2,
              width: mappedBoundingBox.width,
              height: mappedBoundingBox.height,
            },
            confidence: topScore,
            label: topResultScore.index,
          }
          detections.push(detection)
        }
      }
    }
    return detections
  }

  public static filterBoundingBoxes(
    detections: Detection[],
    limit: number,
    threshold: number,
  ) {
    let activeCount = detections.length
    const isActiveBoxes = Array.from({ length: detections.length }, () => true)
    const sortedBoxes = detections
      .map((b, i) => {
        return { box: b, index: i }
      })
      .sort((a, b) => b.box.confidence - a.box.confidence)
    const results = []
    for (let i = 0; i < detections.length; i++) {
      if (isActiveBoxes[i]) {
        const boxA = sortedBoxes[i].box
        results.push(boxA)
        if (results.length >= limit) {
          break
        }
        for (let j = i + 1; j < detections.length; j++) {
          if (isActiveBoxes[j]) {
            var boxB = sortedBoxes[j].box

            if (
              this.intersectionOverUnion(boxA.boundingBox, boxB.boundingBox) >
              threshold
            ) {
              isActiveBoxes[j] = false
              activeCount--

              if (activeCount <= 0) break
            }
          }
        }
        if (activeCount <= 0) break
      }
    }
    return results
  }
}

export { YoloOutputParser }
