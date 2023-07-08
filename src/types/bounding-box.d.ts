type BoundingBox = {
  x: number
  y: number
  width: number
  height: number
}

type Detection = {
  boundingBox: BoundingBox
  confidence: number
  label: number
}
