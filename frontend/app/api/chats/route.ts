import { NextResponse } from "next/server"
import { getSessionUser } from "@/lib/auth"
import { getUserConversations, saveUserConversations, deleteUserConversation } from "@/lib/db"

export async function GET(req: Request) {
  try {
    const session = await getSessionUser(req)
    if (!session) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 })
    }

    const conversations = await getUserConversations(session.id)
    return NextResponse.json({ success: true, conversations })
  } catch (error: any) {
    console.error("[Get Chats Error]:", error)
    return NextResponse.json(
      { error: error?.message || "Failed to load chats from Neon database" },
      { status: 500 }
    )
  }
}

export async function POST(req: Request) {
  try {
    const session = await getSessionUser(req)
    if (!session) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 })
    }

    const body = await req.json()
    const { conversations } = body

    if (!Array.isArray(conversations)) {
      return NextResponse.json(
        { error: "Expected 'conversations' array in request body" },
        { status: 400 }
      )
    }

    await saveUserConversations(session.id, conversations)
    return NextResponse.json({ success: true, count: conversations.length })
  } catch (error: any) {
    console.error("[Save Chats Error]:", error)
    return NextResponse.json(
      { error: error?.message || "Failed to save chats to Neon database" },
      { status: 500 }
    )
  }
}

export async function DELETE(req: Request) {
  try {
    const session = await getSessionUser(req)
    if (!session) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 })
    }

    const { searchParams } = new URL(req.url)
    const id = searchParams.get("id")

    if (!id) {
      return NextResponse.json({ error: "Conversation ID required" }, { status: 400 })
    }

    await deleteUserConversation(session.id, id)
    return NextResponse.json({ success: true })
  } catch (error: any) {
    console.error("[Delete Chat Error]:", error)
    return NextResponse.json(
      { error: error?.message || "Failed to delete chat" },
      { status: 500 }
    )
  }
}
