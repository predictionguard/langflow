import React, { forwardRef } from "react";
import SvgPredictionGuard from "./PredictionGuard";

export const PredictionGuardIcon = forwardRef<
  SVGSVGElement,
  React.PropsWithChildren<{}>
>((props, ref) => {
  return <SvgPredictionGuard ref={ref} {...props} />;
});
