class FPSCounter {
  private readonly _frequency: number
  private _iterator: number
  private _lastUpdateTime: number
  private _callback: (fps: string) => void

  constructor(frequency: number, callback: (fps: string) => void) {
    this._frequency = frequency
    this._iterator = 0
    this._lastUpdateTime = performance.now()
    this._callback = callback
  }

  public click = async () => {
    if (this._iterator % this._frequency === 0) {
      const currentTime = performance.now()
      const duration = (currentTime - this._lastUpdateTime) / 1_000
      this._lastUpdateTime = currentTime
      this._callback((this._frequency / duration).toFixed(2))
    }
    this._iterator++
  }
}

export { FPSCounter }
