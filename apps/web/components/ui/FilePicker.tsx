"use client";

// Choosing files.
//
// The browser's own file control puts a small grey button and the words "no
// file chosen" on the most important action of the page. This replaces it
// with a real button and an area that accepts a folder dragged onto it —
// which is how somebody with thirty photographs actually has them, and it
// spares them the file dialog entirely.
//
// The input is still a plain file input underneath. It is only visually
// hidden, so it keeps its keyboard behaviour and its ref, and the same change
// handler runs whichever way the files arrived.

import { useState, type ChangeEvent, type DragEvent, type RefObject } from "react";
import { controlClass, type ControlRank } from "@/components/ui/Button";

export function FilePicker({
  label,
  hint,
  dropHint,
  accept,
  multiple = false,
  rank = "grey",
  inputRef,
  onFiles,
}: {
  label: string;
  hint: string;
  /** Shown only where there is a mouse. Nobody drags a file on a phone. */
  dropHint?: string;
  accept: string;
  multiple?: boolean;
  rank?: ControlRank;
  inputRef: RefObject<HTMLInputElement | null>;
  onFiles: (files: FileList | null) => void;
}) {
  const [over, setOver] = useState(false);

  function drop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setOver(false);
    onFiles(event.dataTransfer.files);
  }

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={drop}
      className={`flex flex-col items-center gap-4 rounded-2xl border border-dashed px-6 py-10
                  text-center transition-colors duration-200
                  has-[:focus-visible]:outline has-[:focus-visible]:outline-2
                  has-[:focus-visible]:outline-offset-2 has-[:focus-visible]:outline-accent
                  ${over ? "border-accent bg-surface-secondary" : "border-separator-opaque"}`}
    >
      <label className={controlClass({ rank })}>
        <input
          ref={inputRef}
          type="file"
          accept={accept}
          multiple={multiple}
          onChange={(e: ChangeEvent<HTMLInputElement>) => onFiles(e.target.files)}
          className="sr-only"
        />
        {label}
      </label>
      <p className="text-footnote text-label-secondary">
        {hint}
        {dropHint && <span className="hidden [@media(pointer:fine)]:inline"> {dropHint}</span>}
      </p>
    </div>
  );
}
