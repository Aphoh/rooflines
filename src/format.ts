export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const unit = 1024;
  const sizes = ["B", "KB", "MB", "GB", "TB", "PB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(unit)), sizes.length - 1);
  const value = bytes / unit ** index;
  return `${Number(value.toFixed(2))} ${sizes[index]}`;
}

export function formatNumber(value: number): string {
  if (!Number.isFinite(value)) return "-";
  return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

export function formatMetric(value: number | undefined, suffix = ""): string {
  if (value === undefined || value === null || !Number.isFinite(value)) return "-";
  return `${Number(value.toFixed(2)).toLocaleString()}${suffix}`;
}

export function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
