// All data comes from the FastAPI service. Change the URL with NEXT_PUBLIC_API_URL if the API runs elsewhere.
export const API = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export async function get(path) {
  const res = await fetch(`${API}${path}`);
  if (!res.ok) throw new Error(`${path} returned ${res.status}`);
  return res.json();
}
