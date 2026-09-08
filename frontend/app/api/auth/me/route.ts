import { NextResponse } from "next/server"
import { getSessionUser } from "@/lib/auth"
import { findUserById } from "@/lib/db"

export async function GET(req: Request) {
  try {
    const session = await getSessionUser(req)
    if (!session) {
      return NextResponse.json({ authenticated: false, user: null })
    }

    // Verify user still exists in database
    const user = await findUserById(session.id)
    if (!user) {
      return NextResponse.json({ authenticated: false, user: null })
    }

    return NextResponse.json({
      authenticated: true,
      user: {
        id: user.id,
        email: user.email,
        name: user.name,
        avatarId: user.avatar_id || 1,
        createdAt: user.created_at,
      },
    })
  } catch (error: any) {
    console.error("[Auth Me Error]:", error)
    return NextResponse.json(
      { error: error?.message || "Failed to get session" },
      { status: 500 }
    )
  }
}
