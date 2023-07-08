class MiddlewareStream extends TransformStream {
  constructor(callback: () => Promise<void>) {
    super({
      start() {},
      transform(chunk: any, controller: any) {
        callback()
        controller.enqueue(chunk)
      },
      flush() {},
    })
  }
}

export { MiddlewareStream }
