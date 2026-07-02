import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
    const body = await req.json();

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 60000); // 60s timeout
    
    const res = await fetch(`${process.env.API_URL}/predict/album`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "X-API-Key": process.env.TOKEN_API_KEY ?? "",
        },
        body: JSON.stringify(body),
        cache: "no-store",
        signal: controller.signal,
    });
    
    clearTimeout(timeoutId);

    const text = await res.text();
    return new NextResponse(text, {
        status: res.status,
        headers: { "Content-Type": res.headers.get("Content-Type") ?? "application/json" },
    });
}