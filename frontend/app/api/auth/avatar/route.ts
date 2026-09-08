import { NextResponse } from "next/server"
import { getSessionUser } from "@/lib/auth"
import { updateUserAvatar } from "@/lib/db"

export async function POST(req: Request) {
  try {
    const session = await getSessionUser(req)
    if (!session) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 })
    }

    const body = await req.json()
    const avatarId = Number(body.avatarId) || 1

    if (avatarId < 1 || avatarId > 4) {
      return NextResponse.json({ error: "Invalid avatar ID" }, { status: 400 })
    }

    const updated = await updateUserAvatar(session.id, avatarId)
    return NextResponse.json({ success: true, avatarId: updated.avatar_id })
  } catch (err: any) {
    console.error("[Avatar Update Error]:", err)
    return NextResponse.json(
      { error: err?.message || "Failed to update avatar" },
      { status: 500 }
    )
  }
}
