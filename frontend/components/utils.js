export const cls = (...c) => c.filter(Boolean).join(" ");

export function timeAgo(date) {
  if (!date) return "Just now";
  const d = typeof date === "string" ? new Date(date) : date;
  if (isNaN(d.getTime())) return "Just now";
  const now = new Date();
  const sec = Math.max(0, Math.floor((now - d) / 1000));

  if (sec < 60) {
    return "Just now";
  }
  const minutes = Math.floor(sec / 60);
  if (minutes < 60) {
    return `${minutes}m ago`;
  }
  const hours = Math.floor(minutes / 60);
  if (hours < 24) {
    return `${hours}h ago`;
  }
  const days = Math.floor(hours / 24);
  if (days < 7) {
    return `${days}d ago`;
  }
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

export const makeId = (p) => `${p}${Math.random().toString(36).slice(2, 10)}`;
