import { neon } from "@neondatabase/serverless"

/**
 * Neon PostgreSQL Serverless Client for SatQuery AI
 * Uses connection pooling over HTTP for low latency, zero cold starts, and resilience.
 */
function getDb() {
  const connectionString = process.env.DATABASE_URL
  if (!connectionString) {
    throw new Error("DATABASE_URL is not defined in environment variables.")
  }
  return neon(connectionString)
}

let schemaInitialized = false

/**
 * Ensures required tables (users, conversations) exist in Neon DB.
 * Runs once idempotently.
 */
export async function ensureDatabaseSchema() {
  if (schemaInitialized) return
  const sql = getDb()

  await sql.query(`
    CREATE TABLE IF NOT EXISTS users (
      id TEXT PRIMARY KEY,
      email TEXT UNIQUE NOT NULL,
      name TEXT,
      password_hash TEXT NOT NULL,
      created_at TIMESTAMPTZ DEFAULT NOW(),
      updated_at TIMESTAMPTZ DEFAULT NOW()
    );
  `)

  await sql.query(`
    CREATE TABLE IF NOT EXISTS conversations (
      id TEXT PRIMARY KEY,
      user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
      title TEXT NOT NULL,
      folder TEXT,
      pinned BOOLEAN DEFAULT FALSE,
      messages JSONB NOT NULL DEFAULT '[]'::jsonb,
      updated_at TIMESTAMPTZ DEFAULT NOW()
    );
  `)

  await sql.query(`
    CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id);
  `)

  // Ensure avatar_id column exists
  await sql.query(`
    ALTER TABLE users ADD COLUMN IF NOT EXISTS avatar_id INTEGER DEFAULT 1;
  `)

  schemaInitialized = true
}

export async function findUserByEmail(email: string) {
  await ensureDatabaseSchema()
  const sql = getDb()
  const rows = await sql.query(
    "SELECT id, email, name, password_hash, avatar_id, created_at FROM users WHERE LOWER(email) = LOWER($1) LIMIT 1",
    [email.trim()]
  )
  return rows[0] || null
}

export async function findUserById(id: string) {
  await ensureDatabaseSchema()
  const sql = getDb()
  const rows = await sql.query(
    "SELECT id, email, name, avatar_id, created_at FROM users WHERE id = $1 LIMIT 1",
    [id]
  )
  return rows[0] || null
}

export async function createUser({
  id,
  email,
  name,
  passwordHash,
  avatarId = 1,
}: {
  id: string
  email: string
  name: string
  passwordHash: string
  avatarId?: number
}) {
  await ensureDatabaseSchema()
  const sql = getDb()
  const rows = await sql.query(
    `INSERT INTO users (id, email, name, password_hash, avatar_id, created_at, updated_at)
     VALUES ($1, LOWER($2), $3, $4, $5, NOW(), NOW())
     RETURNING id, email, name, avatar_id, created_at`,
    [id, email.trim(), name.trim(), passwordHash, avatarId]
  )
  return rows[0]
}

export async function updateUserAvatar(userId: string, avatarId: number) {
  await ensureDatabaseSchema()
  const sql = getDb()
  const rows = await sql.query(
    `UPDATE users SET avatar_id = $1, updated_at = NOW() WHERE id = $2 RETURNING id, avatar_id`,
    [avatarId, userId]
  )
  return rows[0] || null
}

export async function getUserConversations(userId: string) {
  await ensureDatabaseSchema()
  const sql = getDb()
  const rows = await sql.query(
    `SELECT id, title, folder, pinned, messages, updated_at
     FROM conversations
     WHERE user_id = $1
     ORDER BY updated_at DESC`,
    [userId]
  )

  return rows.map((row: any) => ({
    id: row.id,
    title: row.title,
    folder: row.folder || null,
    pinned: Boolean(row.pinned),
    updatedAt: row.updated_at ? new Date(row.updated_at).toISOString() : new Date().toISOString(),
    messageCount: Array.isArray(row.messages) ? row.messages.length : 0,
    preview: Array.isArray(row.messages) && row.messages.length > 0
      ? String(row.messages[row.messages.length - 1]?.content || "").slice(0, 100)
      : "",
    messages: Array.isArray(row.messages) ? row.messages : [],
  }))
}

export async function saveUserConversations(userId: string, conversations: any[]) {
  await ensureDatabaseSchema()
  const sql = getDb()

  for (const conv of conversations) {
    if (!conv || !conv.id) continue

    // Sanitize heavy base64 data to preserve database efficiency
    const sanitizedMessages = (conv.messages || []).map((msg: any) => ({
      ...msg,
      apiFiles: (msg.apiFiles || []).map((file: any) => ({
        name: file.name,
        previewUrl: file.previewUrl || "",
        b64: file.b64 && file.b64.length < 100000 ? file.b64 : undefined,
      })),
    }))

    const clientTimestamp = conv.updatedAt && !isNaN(new Date(conv.updatedAt).getTime())
      ? new Date(conv.updatedAt).toISOString()
      : null

    await sql.query(
      `INSERT INTO conversations (id, user_id, title, folder, pinned, messages, updated_at)
       VALUES ($1, $2, $3, $4, $5, $6::jsonb, COALESCE($7::timestamptz, NOW()))
       ON CONFLICT (id) DO UPDATE SET
         title = EXCLUDED.title,
         folder = EXCLUDED.folder,
         pinned = EXCLUDED.pinned,
         messages = EXCLUDED.messages,
         updated_at = COALESCE($7::timestamptz, conversations.updated_at, NOW())`,
      [
        conv.id,
        userId,
        conv.title || "Satellite Query",
        conv.folder || null,
        Boolean(conv.pinned),
        JSON.stringify(sanitizedMessages),
        clientTimestamp,
      ]
    )
  }

  return true
}

export async function deleteUserConversation(userId: string, conversationId: string) {
  await ensureDatabaseSchema()
  const sql = getDb()
  await sql.query(
    "DELETE FROM conversations WHERE id = $1 AND user_id = $2",
    [conversationId, userId]
  )
  return true
}
