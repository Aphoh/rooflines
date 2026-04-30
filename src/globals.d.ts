declare const Bun: any;
declare const process: any;

declare module "bun:test" {
  export const describe: any;
  export const expect: any;
  export const test: any;
}
